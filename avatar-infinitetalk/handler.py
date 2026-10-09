import runpod
import os
import time
import ipaddress
import socket
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii  # Base64 에러 처리를 위해 import
import subprocess
import librosa
import shutil

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def truncate_base64_for_log(base64_str, max_length=50):
    """Base64 문자열을 로깅용으로 앞부분만 잘라서 반환"""
    if not base64_str:
        return "None"
    if len(base64_str) <= max_length:
        return base64_str
    return f"{base64_str[:max_length]}... (총 {len(base64_str)} 문자)"


server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Security hardening (2026-10-09). job["input"] is attacker-reachable: anyone
# holding the RunPod API key can post arbitrary fields. Three defences added
# here: SSRF/redirect-SSRF on url inputs, argument injection on the wget
# command line, and path traversal / arbitrary file read on path inputs.
# Also: a size cap on decoded base64 and bounds on numeric fields so a bad
# job can't exhaust disk/RAM or run up GPU time by itself.
# ---------------------------------------------------------------------------

_ALLOWED_URL_SCHEMES = {"http", "https"}
_MAX_INPUT_BYTES = 100 * 1024 * 1024  # 100 MB -- generous for one image/audio clip
SANDBOX_ROOT = os.path.realpath("/tmp/infinitetalk_io")
os.makedirs(SANDBOX_ROOT, exist_ok=True)


def _check_ip_not_internal(ip_str, hostname):
    ip_str = ip_str.split("%", 1)[0]  # strip an IPv6 zone id if present
    ip = ipaddress.ip_address(ip_str)
    mapped = ip.ipv4_mapped if hasattr(ip, "ipv4_mapped") else None
    if mapped is not None:
        ip = mapped
    if (ip.is_private or ip.is_loopback or ip.is_link_local or
            ip.is_multicast or ip.is_reserved or ip.is_unspecified):
        raise Exception(
            f"거부됨(SSRF 방지): {hostname}가 내부/사설 주소로 해석됩니다: {ip_str}"
        )


def validate_external_url(url):
    """http(s)만 허용하고, 호스트가 사설/루프백/링크로컬/예약 대역으로 해석되면
    거부한다 (SSRF 방지). 인자 주입 방지를 위해 '-'로 시작하는 값도 거부한다."""
    if not isinstance(url, str) or not url.strip():
        raise Exception("URL이 비어 있습니다")
    if url.lstrip().startswith("-"):
        raise Exception("URL이 '-'로 시작할 수 없습니다 (인자 주입 방지)")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        raise Exception(f"지원하지 않는 URL 스킴입니다: {parsed.scheme!r}")
    hostname = parsed.hostname
    if not hostname:
        raise Exception("URL에 호스트가 없습니다")
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise Exception(f"DNS 확인 실패: {hostname}: {e}")
    if not infos:
        raise Exception(f"DNS 확인 결과가 없습니다: {hostname}")
    for info in infos:
        _check_ip_not_internal(info[4][0], hostname)
    return url


def _resolve_within_sandbox(path_str):
    """로컬 경로 입력은 전용 샌드박스 디렉터리 내부로만 해석한다 (경로 탈출 /
    임의 파일 읽기 방지). realpath로 심볼릭 링크까지 풀어서 접두사를 검사한다."""
    if not isinstance(path_str, str) or not path_str.strip():
        raise Exception("경로가 비어 있습니다")
    relative = path_str.lstrip("/\\")
    candidate = os.path.realpath(os.path.join(SANDBOX_ROOT, relative))
    root_with_sep = SANDBOX_ROOT + os.sep
    if candidate != SANDBOX_ROOT and not candidate.startswith(root_with_sep):
        raise Exception(f"샌드박스 밖의 경로는 허용되지 않습니다: {path_str}")
    if not os.path.isfile(candidate):
        raise Exception(f"샌드박스 내에 파일이 없습니다: {path_str}")
    return candidate


def _safe_temp_dir(temp_dir):
    """temp_dir을 항상 SANDBOX_ROOT 하위로 강제한다 (호출자가 무엇을 넘기든)."""
    return os.path.join(SANDBOX_ROOT, os.path.basename(str(temp_dir)) or "job")


def clamp_int(value, default, lo, hi, name):
    """정수가 아니거나 허용 범위를 벗어나면 기본값으로 되돌린다 (리소스 고갈/
    과금 남용 방지: width·height·max_frame은 job_input에서 그대로 온다)."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        logger.warning(f"⚠️ {name} 값이 올바르지 않습니다({value!r}); 기본값 {default} 사용")
        return default
    if value < lo or value > hi:
        logger.warning(f"⚠️ {name}={value}가 허용 범위[{lo},{hi}]를 벗어나 기본값 {default}로 보정합니다")
        return default
    return value


def download_file_from_url(url, output_path):
    """URL에서 파일을 다운로드하는 함수"""
    validate_external_url(url)
    try:
        # wget을 사용하여 파일 다운로드.
        # --max-redirect=0: 리다이렉트를 따라가지 않는다 -- 공인 URL이 사설
        #   주소로 302를 보내는 SSRF 우회를 막는 가장 확실한 방법.
        # --tries=1, --quota=100m: 재시도/다운로드 크기를 제한해 디스크 고갈을 막는다.
        # "--" 뒤에 url: url이 '-'로 시작해도 wget 옵션으로 해석되지 않는다
        #   (인자 주입 방지, validate_external_url의 검사와 이중 방어).
        result = subprocess.run(
            ["wget", "-O", output_path, "--no-verbose", "--timeout=30",
             "--tries=1", "--max-redirect=0", "--quota=100m", "--", url],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            logger.info(
                f"✅ URL에서 파일을 성공적으로 다운로드했습니다: {url} -> {output_path}"
            )
            return output_path
        else:
            logger.error(f"❌ wget 다운로드 실패: {result.stderr}")
            raise Exception(f"URL 다운로드 실패: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("❌ 다운로드 시간 초과")
        raise Exception("다운로드 시간 초과")
    except Exception as e:
        logger.error(f"❌ 다운로드 중 오류 발생: {e}")
        raise Exception(f"다운로드 중 오류 발생: {e}")


def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Base64 데이터를 파일로 저장하는 함수"""
    try:
        if not isinstance(base64_data, str) or not base64_data:
            raise Exception("base64 입력이 비어 있습니다")
        # 디코딩 전에 인코딩 길이로 상한을 먼저 거른다 (디코딩 자체가 메모리를
        # 쓰기 전에 거대한 페이로드를 거부하기 위함).
        if len(base64_data) > (_MAX_INPUT_BYTES * 4 // 3 + 8):
            raise Exception(f"base64 입력이 너무 큽니다 ({len(base64_data)} 문자)")

        # Base64 문자열 디코딩
        decoded_data = base64.b64decode(base64_data, validate=True)
        if len(decoded_data) > _MAX_INPUT_BYTES:
            raise Exception(f"디코딩된 파일이 너무 큽니다 ({len(decoded_data)} bytes)")

        # 디렉토리가 존재하지 않으면 생성 (항상 샌드박스 하위)
        safe_dir = _safe_temp_dir(temp_dir)
        os.makedirs(safe_dir, exist_ok=True)

        # 파일로 저장
        file_path = os.path.join(safe_dir, output_filename)
        with open(file_path, "wb") as f:
            f.write(decoded_data)

        logger.info(f"✅ Base64 입력을 '{file_path}' 파일로 저장했습니다.")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"❌ Base64 디코딩 실패: {e}")
        raise Exception(f"Base64 디코딩 실패: {e}")


def process_input(input_data, temp_dir, output_filename, input_type):
    """입력 데이터를 처리하여 파일 경로를 반환하는 함수"""
    if input_type == "path":
        # 경로는 전용 샌드박스 내부에서만 해석한다 (경로 탈출/임의 파일 읽기 방지)
        logger.info(f"📁 경로 입력 처리: {input_data}")
        return _resolve_within_sandbox(input_data)
    elif input_type == "url":
        # URL인 경우 다운로드 (SSRF 방지 검증은 download_file_from_url에서 수행)
        logger.info(f"🌐 URL 입력 처리: {input_data}")
        safe_dir = _safe_temp_dir(temp_dir)
        os.makedirs(safe_dir, exist_ok=True)
        file_path = os.path.join(safe_dir, output_filename)
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        # Base64인 경우 디코딩하여 저장
        logger.info(f"🔢 Base64 입력 처리")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"지원하지 않는 입력 타입: {input_type}")


def queue_prompt(prompt, input_type="image", person_count="single"):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")

    # 디버깅을 위해 워크플로우 내용 로깅
    logger.info(f"워크플로우 노드 수: {len(prompt)}")
    if input_type == "image":
        logger.info(
            f"이미지 노드(284) 설정: {prompt.get('284', {}).get('inputs', {}).get('image', 'NOT_FOUND')}"
        )
    else:
        logger.info(
            f"비디오 노드(228) 설정: {prompt.get('228', {}).get('inputs', {}).get('video', 'NOT_FOUND')}"
        )
    logger.info(
        f"오디오 노드(125) 설정: {prompt.get('125', {}).get('inputs', {}).get('audio', 'NOT_FOUND')}"
    )
    logger.info(
        f"텍스트 노드(241) 설정: {prompt.get('241', {}).get('inputs', {}).get('positive_prompt', 'NOT_FOUND')}"
    )
    if person_count == "multi":
        if "307" in prompt:
            logger.info(
                f"두 번째 오디오 노드(307) 설정: {prompt.get('307', {}).get('inputs', {}).get('audio', 'NOT_FOUND')}"
            )
        elif "313" in prompt:
            logger.info(
                f"두 번째 오디오 노드(313) 설정: {prompt.get('313', {}).get('inputs', {}).get('audio', 'NOT_FOUND')}"
            )

    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", "application/json")

    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        logger.info(f"프롬프트 전송 성공: {result}")
        return result
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP 에러 발생: {e.code} - {e.reason}")
        logger.error(f"응답 내용: {e.read().decode('utf-8')}")
        raise
    except Exception as e:
        logger.error(f"프롬프트 전송 중 오류: {e}")
        raise


def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()


def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def get_videos(ws, prompt, input_type="image", person_count="single"):
    prompt_id = queue_prompt(prompt, input_type, person_count)["prompt_id"]
    logger.info(f"워크플로우 실행 시작: prompt_id={prompt_id}")

    output_videos = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is not None:
                    logger.info(f"노드 실행 중: {data['node']}")
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    logger.info("워크플로우 실행 완료")
                    break
        else:
            continue

    logger.info(f"히스토리 조회 중: prompt_id={prompt_id}")
    history = get_history(prompt_id)[prompt_id]
    logger.info(f"출력 노드 수: {len(history['outputs'])}")

    for node_id in history["outputs"]:
        node_output = history["outputs"][node_id]
        videos_output = []
        if "gifs" in node_output:
            logger.info(
                f"노드 {node_id}에서 {len(node_output['gifs'])}개의 비디오 발견"
            )
            for idx, video in enumerate(node_output["gifs"]):
                # fullpath를 그대로 반환 (base64 인코딩하지 않음)
                video_path = video["fullpath"]
                logger.info(f"비디오 파일 경로: {video_path}")

                # 파일 존재 여부 및 크기 확인
                if os.path.exists(video_path):
                    file_size = os.path.getsize(video_path)
                    logger.info(
                        f"비디오 {idx+1} 발견: {video_path} (크기: {file_size} bytes)"
                    )
                else:
                    logger.warning(f"비디오 파일이 존재하지 않습니다: {video_path}")

                videos_output.append(video_path)
        else:
            logger.info(f"노드 {node_id}에 비디오 출력 없음")
        output_videos[node_id] = videos_output

    logger.info(f"총 {len(output_videos)}개 노드에서 비디오 파일 경로 수집 완료")
    return output_videos


def load_workflow(workflow_path):
    with open(workflow_path, "r") as file:
        return json.load(file)


def get_workflow_path(input_type, person_count):
    """input_type과 person_count에 따라 적절한 워크플로우 파일 경로를 반환"""
    if input_type == "image":
        if person_count == "single":
            return "/I2V_single.json"
        else:  # multi
            return "/I2V_multi.json"
    else:  # video
        if person_count == "single":
            return "/V2V_single.json"
        else:  # multi
            return "/V2V_multi.json"


def get_audio_duration(audio_path):
    """오디오 파일의 길이(초)를 반환"""
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        logger.warning(f"오디오 길이 계산 실패 ({audio_path}): {e}")
        return None


def calculate_max_frames_from_audio(wav_path, wav_path_2=None, fps=25):
    """오디오 길이를 기반으로 max_frames를 계산"""
    durations = []

    # 첫 번째 오디오 길이 계산
    duration1 = get_audio_duration(wav_path)
    if duration1 is not None:
        durations.append(duration1)
        logger.info(f"첫 번째 오디오 길이: {duration1:.2f}초")

    # 두 번째 오디오 길이 계산 (multi person인 경우)
    if wav_path_2:
        duration2 = get_audio_duration(wav_path_2)
        if duration2 is not None:
            durations.append(duration2)
            logger.info(f"두 번째 오디오 길이: {duration2:.2f}초")

    if not durations:
        logger.warning("오디오 길이를 계산할 수 없습니다. 기본값 81을 사용합니다.")
        return 81

    # 가장 긴 오디오 길이를 기준으로 max_frames 계산
    max_duration = max(durations)
    max_frames = int(max_duration * fps) + 81

    logger.info(
        f"가장 긴 오디오 길이: {max_duration:.2f}초, 계산된 max_frames: {max_frames}"
    )
    return max_frames


def handler(job):
    job_input = job.get("input", {})

    # job_input을 로깅할 때 base64 데이터는 truncate해서 출력
    log_input = job_input.copy()
    for key in ["image_base64", "video_base64", "wav_base64", "wav_base64_2"]:
        if key in log_input:
            log_input[key] = truncate_base64_for_log(log_input[key])

    logger.info(f"Received job input: {log_input}")
    task_id = f"task_{uuid.uuid4()}"

    # 입력 타입과 인물 수 확인
    input_type = job_input.get("input_type", "image")  # "image" 또는 "video"
    person_count = job_input.get("person_count", "single")  # "single" 또는 "multi"

    logger.info(f"워크플로우 타입: {input_type}, 인물 수: {person_count}")

    # 워크플로우 파일 경로 결정
    workflow_path = get_workflow_path(input_type, person_count)
    logger.info(f"사용할 워크플로우: {workflow_path}")

    # 이미지/비디오 입력 처리
    media_path = None
    if input_type == "image":
        # 이미지 입력 처리 (image_path, image_url, image_base64 중 하나만 사용)
        if "image_path" in job_input:
            media_path = process_input(
                job_input["image_path"], task_id, "input_image.jpg", "path"
            )
        elif "image_url" in job_input:
            media_path = process_input(
                job_input["image_url"], task_id, "input_image.jpg", "url"
            )
        elif "image_base64" in job_input:
            media_path = process_input(
                job_input["image_base64"], task_id, "input_image.jpg", "base64"
            )
        else:
            # 기본값 사용
            media_path = "/examples/image.jpg"
            logger.info("기본 이미지 파일을 사용합니다: /examples/image.jpg")
    else:  # video
        # 비디오 입력 처리 (video_path, video_url, video_base64 중 하나만 사용)
        if "video_path" in job_input:
            media_path = process_input(
                job_input["video_path"], task_id, "input_video.mp4", "path"
            )
        elif "video_url" in job_input:
            media_path = process_input(
                job_input["video_url"], task_id, "input_video.mp4", "url"
            )
        elif "video_base64" in job_input:
            media_path = process_input(
                job_input["video_base64"], task_id, "input_video.mp4", "base64"
            )
        else:
            # 기본값 사용 (비디오가 없는 경우 기본 이미지 사용)
            media_path = "/examples/image.jpg"
            logger.info("기본 이미지 파일을 사용합니다: /examples/image.jpg")

    # 오디오 입력 처리 (wav_path, wav_url, wav_base64 중 하나만 사용)
    wav_path = None
    wav_path_2 = None  # 다중 인물용 두 번째 오디오

    if "wav_path" in job_input:
        wav_path = process_input(
            job_input["wav_path"], task_id, "input_audio.wav", "path"
        )
    elif "wav_url" in job_input:
        wav_path = process_input(
            job_input["wav_url"], task_id, "input_audio.wav", "url"
        )
    elif "wav_base64" in job_input:
        wav_path = process_input(
            job_input["wav_base64"], task_id, "input_audio.wav", "base64"
        )
    else:
        # 기본값 사용
        wav_path = "/examples/audio.mp3"
        logger.info("기본 오디오 파일을 사용합니다: /examples/audio.mp3")

    # 다중 인물용 두 번째 오디오 처리
    if person_count == "multi":
        if "wav_path_2" in job_input:
            wav_path_2 = process_input(
                job_input["wav_path_2"], task_id, "input_audio_2.wav", "path"
            )
        elif "wav_url_2" in job_input:
            wav_path_2 = process_input(
                job_input["wav_url_2"], task_id, "input_audio_2.wav", "url"
            )
        elif "wav_base64_2" in job_input:
            wav_path_2 = process_input(
                job_input["wav_base64_2"], task_id, "input_audio_2.wav", "base64"
            )
        else:
            # 기본값 사용 (첫 번째 오디오와 동일)
            wav_path_2 = wav_path
            logger.info("두 번째 오디오가 없어 첫 번째 오디오를 사용합니다.")

    # 필수 필드 검증 및 기본값 설정
    prompt_text = job_input.get("prompt", "A person talking naturally")
    # 리소스 고갈/과금 남용 방지: 해상도는 64~1536 범위로 고정
    width = clamp_int(job_input.get("width", 512), 512, 64, 1536, "width")
    height = clamp_int(job_input.get("height", 512), 512, 64, 1536, "height")

    # max_frame 설정 (입력이 없으면 오디오 길이 기반으로 자동 계산)
    max_frame = job_input.get("max_frame")
    if max_frame is None:
        logger.info(
            "max_frame이 입력되지 않았습니다. 오디오 길이를 기반으로 자동 계산합니다."
        )
        max_frame = calculate_max_frames_from_audio(
            wav_path, wav_path_2 if person_count == "multi" else None
        )
    else:
        logger.info(f"사용자 지정 max_frame: {max_frame}")
    # 2000 frames @25fps = 80초 -- 실사용을 훨씬 웃도는 상한이지만, 입력이
    # 자동 계산이든 사용자 지정이든 과도한 GPU 시간을 막기 위해 둘 다 고정한다.
    max_frame = clamp_int(max_frame, 121, 9, 2000, "max_frame")

    logger.info(
        f"워크플로우 설정: prompt='{prompt_text}', width={width}, height={height}, max_frame={max_frame}"
    )
    logger.info(f"미디어 경로: {media_path}")
    logger.info(f"오디오 경로: {wav_path}")
    if person_count == "multi":
        logger.info(f"두 번째 오디오 경로: {wav_path_2}")

    prompt = load_workflow(workflow_path)

    # ------------------------------------------------------------------
    # 동적 Force Offload 설정
    # ------------------------------------------------------------------
    # 1. 입력에서 force_offload 읽기 (기본값 True: 작은 GPU에서 OOM 방지)
    force_offload = bool(job_input.get("force_offload", True))
    logger.info(f"🔧 설정: force_offload={force_offload}")

    # 2. WanVideoSampler 노드에 force_offload 파라미터 주입
    sampler_node_id = None
    preferred_id = "128"

    # 효율성을 위해 먼저 선호 ID(128) 확인
    if preferred_id in prompt and prompt[preferred_id].get("class_type") == "WanVideoSampler":
        sampler_node_id = preferred_id
    else:
        # ID가 다른 경우 class type으로 검색 (폴백)
        for node_id, node_data in prompt.items():
            if node_data.get("class_type") == "WanVideoSampler":
                sampler_node_id = node_id
                break

    # sampler 노드를 찾은 경우 force_offload 파라미터 주입
    if sampler_node_id:
        # setdefault를 사용하여 'inputs' 딕셔너리가 없으면 생성
        inputs = prompt[sampler_node_id].setdefault("inputs", {})
        inputs["force_offload"] = force_offload
        logger.info(f"✅ 노드 {sampler_node_id} (WanVideoSampler) 업데이트됨: force_offload={force_offload}")
    else:
        logger.warning("⚠️ 경고: WanVideoSampler 노드를 찾을 수 없습니다. 워크플로우 기본값을 사용합니다.")
    # ------------------------------------------------------------------

    # 파일 존재 여부 확인
    if not os.path.exists(media_path):
        logger.error(f"미디어 파일이 존재하지 않습니다: {media_path}")
        return {"error": f"미디어 파일을 찾을 수 없습니다: {media_path}"}

    if not os.path.exists(wav_path):
        logger.error(f"오디오 파일이 존재하지 않습니다: {wav_path}")
        return {"error": f"오디오 파일을 찾을 수 없습니다: {wav_path}"}

    if person_count == "multi" and wav_path_2 and not os.path.exists(wav_path_2):
        logger.error(f"두 번째 오디오 파일이 존재하지 않습니다: {wav_path_2}")
        return {"error": f"두 번째 오디오 파일을 찾을 수 없습니다: {wav_path_2}"}

    logger.info(f"미디어 파일 크기: {os.path.getsize(media_path)} bytes")
    logger.info(f"오디오 파일 크기: {os.path.getsize(wav_path)} bytes")
    if person_count == "multi" and wav_path_2:
        logger.info(f"두 번째 오디오 파일 크기: {os.path.getsize(wav_path_2)} bytes")

    # 워크플로우 노드 설정
    if input_type == "image":
        # I2V 워크플로우: 이미지 입력 설정
        prompt["284"]["inputs"]["image"] = media_path
    else:
        # V2V 워크플로우: 비디오 입력 설정
        prompt["228"]["inputs"]["video"] = media_path

    # 공통 설정
    prompt["125"]["inputs"]["audio"] = wav_path
    prompt["241"]["inputs"]["positive_prompt"] = prompt_text
    prompt["245"]["inputs"]["value"] = width
    prompt["246"]["inputs"]["value"] = height

    prompt["270"]["inputs"]["value"] = max_frame

    # 다중 인물용 두 번째 오디오 설정
    if person_count == "multi":
        # 워크플로우 타입에 따라 두 번째 오디오 노드 설정
        if input_type == "image":  # I2V_multi.json의 경우
            if "307" in prompt:
                prompt["307"]["inputs"]["audio"] = wav_path_2
        else:  # V2V_multi.json의 경우
            if "313" in prompt:
                prompt["313"]["inputs"]["audio"] = wav_path_2

    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")

    # 먼저 HTTP 연결이 가능한지 확인
    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")

    # HTTP 연결 확인 (최대 1분)
    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            import urllib.request

            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP 연결 성공 (시도 {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(
                f"HTTP 연결 실패 (시도 {http_attempt+1}/{max_http_attempts}): {e}"
            )
            if http_attempt == max_http_attempts - 1:
                raise Exception(
                    "ComfyUI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요."
                )
            time.sleep(1)

    ws = websocket.WebSocket()
    # 웹소켓 연결 시도 (최대 3분)
    max_attempts = int(180 / 5)  # 3분 (1초에 한 번씩 시도)
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"웹소켓 연결 성공 (시도 {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"웹소켓 연결 실패 (시도 {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                raise Exception("웹소켓 연결 시간 초과 (3분)")
            time.sleep(5)
    videos = get_videos(ws, prompt, input_type, person_count)
    ws.close()
    logger.info("웹소켓 연결 종료")

    # 비디오가 없는 경우 처리
    output_video_path = None
    logger.info("출력 비디오 검색 중...")

    for node_id in videos:
        if videos[node_id]:
            output_video_path = videos[node_id][0]
            logger.info(f"노드 {node_id}에서 출력 비디오 발견: {output_video_path}")
            break
        else:
            logger.info(f"노드 {node_id}는 비어있음")

    if not output_video_path:
        logger.error("출력 비디오를 찾을 수 없습니다. 모든 노드가 비어있습니다.")
        return {"error": "비디오를 찾을 수 없습니다."}

    # 비디오 파일 존재 여부 확인
    if not os.path.exists(output_video_path):
        logger.error(f"출력 비디오 파일이 존재하지 않습니다: {output_video_path}")
        return {"error": f"비디오 파일을 찾을 수 없습니다: {output_video_path}"}

    # network_volume 파라미터 확인
    use_network_volume = job_input.get("network_volume", False)
    logger.info(f"네트워크 볼륨 사용 여부: {use_network_volume}")

    if use_network_volume:
        # 네트워크 볼륨 사용: 파일 복사
        logger.info("네트워크 볼륨에 비디오 복사 시작")
        try:
            # 결과 비디오 파일 경로 생성
            output_filename = f"infinitetalk_{task_id}.mp4"
            output_path = f"/runpod-volume/{output_filename}"
            logger.info(f"원본 파일: {output_video_path}")
            logger.info(f"대상 경로: {output_path}")

            # 원본 파일 크기 확인
            source_file_size = os.path.getsize(output_video_path)
            logger.info(f"원본 파일 크기: {source_file_size} bytes")

            # 파일 복사 (shutil.copy2는 메타데이터도 함께 복사)
            shutil.copy2(output_video_path, output_path)
            logger.info("파일 복사 완료")

            # 복사된 파일 크기 확인
            copied_file_size = os.path.getsize(output_path)
            logger.info(f"복사된 파일 크기: {copied_file_size} bytes")

            if source_file_size == copied_file_size:
                logger.info(
                    f"✅ 결과 비디오를 '{output_path}'에 성공적으로 복사했습니다"
                )
            else:
                logger.warning(
                    f"⚠️ 파일 크기가 일치하지 않습니다: 원본={source_file_size}, 복사본={copied_file_size}"
                )

            return {"video_path": output_path}

        except Exception as e:
            logger.error(f"❌ 비디오 복사 실패: {e}")
            return {"error": f"비디오 복사 실패: {e}"}
    else:
        # 네트워크 볼륨 미사용: Base64 인코딩하여 반환
        logger.info("Base64 인코딩 시작")
        logger.info(f"비디오 파일 경로: {output_video_path}")

        try:
            # 파일 크기 확인
            file_size = os.path.getsize(output_video_path)
            logger.info(f"원본 파일 크기: {file_size} bytes")

            # 파일을 읽어 base64 인코딩
            with open(output_video_path, "rb") as f:
                video_data = base64.b64encode(f.read()).decode("utf-8")

            encoded_size = len(video_data)
            logger.info(f"Base64 인코딩 완료: {encoded_size} 문자")
            logger.info(
                f"✅ Base64 인코딩된 비디오 반환: {truncate_base64_for_log(video_data)}"
            )
            return {"video": video_data}

        except Exception as e:
            logger.error(f"❌ Base64 인코딩 실패: {e}")
            return {"error": f"Base64 인코딩 실패: {e}"}


runpod.serverless.start({"handler": handler})

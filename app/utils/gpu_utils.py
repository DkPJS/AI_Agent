"""GPU 설정 및 유틸리티 함수"""
import torch
import logging
from typing import Tuple, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

def setup_gpu_device() -> torch.device:
    """GPU 디바이스 설정

    Returns:
        torch.device: 사용할 디바이스 (cuda 또는 cpu)
    """
    if settings.FORCE_CPU:
        logger.info("FORCE_CPU 설정으로 인해 CPU를 사용합니다.")
        return torch.device('cpu')

    if not settings.USE_GPU:
        logger.info("USE_GPU 설정이 비활성화되어 CPU를 사용합니다.")
        return torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU 사용: {gpu_name} (메모리: {gpu_memory:.1f}GB)")

        # GPU 메모리 관리 설정
        configure_gpu_memory()

        return device
    else:
        logger.warning("CUDA를 사용할 수 없어 CPU로 fallback합니다.")
        return torch.device('cpu')

def configure_gpu_memory() -> None:
    """GPU 메모리 설정"""
    if torch.cuda.is_available():
        try:
            # PyTorch 메모리 할당 최적화
            torch.cuda.empty_cache()

            # GPU 메모리 제한 해제 (메모리 부족 문제 해결)
            # memory_fraction 설정을 주석 처리하여 PyTorch가 필요한 만큼 메모리 사용하도록 허용
            # memory_fraction = settings.GPU_MEMORY_FRACTION
            # if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            #     torch.cuda.set_per_process_memory_fraction(memory_fraction)
            #     logger.info(f"GPU 메모리 사용률을 {memory_fraction*100}%로 설정")
            logger.info("GPU 메모리 제한을 해제하여 동적 할당 허용")

            # CUDA 메모리 관리 개선
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # 메모리 풀 설정 (fragmentation 방지)
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

            logger.info("GPU 메모리 fragmentation 방지 설정 완료")

        except Exception as e:
            logger.warning(f"GPU 메모리 설정 중 오류: {e}")

def get_gpu_info() -> dict:
    """GPU 정보 반환

    Returns:
        dict: GPU 정보 딕셔너리
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": None,
        "memory_allocated": 0,
        "memory_reserved": 0,
        "memory_total": 0
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
        info["memory_reserved"] = torch.cuda.memory_reserved() / 1024**3    # GB
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

    return info

def optimize_model_for_inference(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """모델을 추론용으로 최적화

    Args:
        model: PyTorch 모델
        device: 대상 디바이스

    Returns:
        torch.nn.Module: 최적화된 모델
    """
    try:
        # 모델을 디바이스로 이동
        model = model.to(device)

        # 평가 모드로 설정
        model.eval()

        # 그래디언트 계산 비활성화
        for param in model.parameters():
            param.requires_grad = False

        # 메모리 정리
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        logger.info(f"모델이 {device}에서 추론용으로 최적화되었습니다.")
        return model

    except Exception as e:
        logger.error(f"모델 최적화 중 오류: {e}")
        raise

def clear_gpu_cache() -> None:
    """GPU 캐시 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU 캐시를 정리했습니다.")

def monitor_gpu_memory() -> Tuple[float, float]:
    """GPU 메모리 사용량 모니터링

    Returns:
        Tuple[float, float]: (사용 중인 메모리 GB, 전체 메모리 GB)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0

    allocated = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return allocated, total

def check_gpu_health() -> bool:
    """GPU 상태 확인

    Returns:
        bool: GPU가 정상적으로 작동하는지 여부
    """
    if not torch.cuda.is_available():
        return False

    try:
        # 간단한 GPU 연산 테스트
        test_tensor = torch.randn(100, 100, device='cuda')
        result = torch.matmul(test_tensor, test_tensor)
        del test_tensor, result
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        logger.error(f"GPU 상태 확인 실패: {e}")
        return False
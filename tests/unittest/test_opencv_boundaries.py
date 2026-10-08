"""守卫无需 OpenCV 的实际路径及模型输入替代操作的数值行为。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from mineru.model.ocr.image import alpha_to_color, check_img, get_rotate_crop_image, rgb_to_bgr
from mineru.model.table.cls.mineru_table_ori_cls import MineruTableOrientationClsModel

_ROOT = Path(__file__).resolve().parents[2]
_BLOCK_CV2 = '''
import importlib.abc
import sys

class BlockCv2(importlib.abc.MetaPathFinder):
    """在实际入口执行前拒绝直接或间接的 OpenCV 导入。"""
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cv2" or fullname.startswith("cv2."):
            raise ImportError("OpenCV must not load on this path")

sys.meta_path.insert(0, BlockCv2())
'''


def _run_blocked(script: str, *args: str) -> None:
    """在独立进程验证真实加载边界，避免其他测试已加载 cv2 掩盖泄漏。"""
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_CV2 + script, *args],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pdf_and_model_helpers_import_without_opencv() -> None:
    """混合编排和模型模块的导入不提前加载 OpenCV 或实际表格模型。"""
    _run_blocked("""
import importlib
for module in (
    "mineru.parser", "mineru.render", "mineru.model.vlm.client",
    "mineru.backend.analysis.pdf.pipeline", "mineru.backend.analysis.pdf.window",
    "mineru.backend.analysis.pdf.formulas", "mineru.backend.analysis.pdf.ocr",
    "mineru.backend.analysis.pdf.tables", "mineru.backend.analysis.pdf.text.content",
    "mineru.model.runtime.hybrid", "mineru.model.ocr.image",
    "mineru.model.ocr.db_postprocess", "mineru.model.ocr.seal_crop",
    "mineru.model.table.cls.paddle_table_cls", "mineru.model.table.rec.unet_table.utils",
    "mineru.model.table.rec.unet_table.utils_table_line_rec",
):
    importlib.import_module(module)
assert "cv2" not in sys.modules
assert "mineru.model.table.rec.unet_table.main" not in sys.modules
assert "mineru.model.table.rec.slanet_plus.main" not in sys.modules
""")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("ocr_mode", ["txt", "auto"])
def test_real_flash_pdf_parse_and_all_renderers_without_opencv(tmp_path: Path, asynchronous: bool, ocr_mode: str) -> None:
    """真实选页、自动分类、原生补图和九种渲染均不加载 OpenCV。"""
    source = tmp_path / "source.pdf"
    canvas = Canvas(str(source))
    for page in range(3):
        for line in range(12):
            canvas.drawString(50, 760 - line * 18, f"Page {page + 1}: Native text evidence line {line}.")
        with Image.new("RGB", (180, 90), (30, 80, 180)) as picture:
            canvas.drawImage(ImageReader(picture), 50, 350, width=180, height=90)
        canvas.showPage()
    canvas.save()
    _run_blocked(
        """
import asyncio
from pathlib import Path
from mineru.config import config
from mineru.parser import parse, parse_async
from mineru.parser.writer import FileBasedDataWriter
from mineru.render import RenderFormat, render

config.llm_aided.features.title_leveling = False
config.llm_aided.features.cross_page_table_cell_merge = False
options = dict(tier="flash", ocr_mode=sys.argv[2], page_range="1,3")
result = asyncio.run(parse_async(sys.argv[1], **options)) if sys.argv[3] == "True" else parse(sys.argv[1], **options)
assert len(result.pages) == 2
assert result.middle_json.extensions["mineru"]["parse_mode"] == "txt"
assert "Page 1" in result.markdown() and "Page 3" in result.markdown()
assert "Page 2" not in result.markdown()
assert "image" in result.middle_json.model_dump_json()
assert "mineru.model.runtime.hybrid" not in sys.modules
before = result.middle_json.model_dump_json()
for output_format in RenderFormat:
    assert render(result.middle_json, output_format)
result.save(FileBasedDataWriter(str(Path(sys.argv[1]).parent / "export")))
assert before == result.middle_json.model_dump_json()
assert "cv2" not in sys.modules
""",
        str(source),
        ocr_mode,
        str(asynchronous),
    )


def test_simple_image_operations_without_opencv() -> None:
    """颜色转换、alpha 合成、直角旋转和轴对齐裁剪不进入 OpenCV。"""
    _run_blocked("""
import numpy as np
from mineru.model.ocr.image import alpha_to_color, check_img, get_rotate_crop_image, rgb_to_bgr
from mineru.model.table.cls.mineru_table_ori_cls import MineruTableOrientationClsModel
image = np.zeros((8, 10, 4), dtype=np.uint8)
assert rgb_to_bgr(image).shape == (8, 10, 3)
assert alpha_to_color(image).shape == (8, 10, 3)
assert check_img(image[:, :, 0]).shape == (8, 10, 3)
points = np.array([[1, 1], [7, 1], [7, 6], [1, 6]], dtype=np.float32)
assert get_rotate_crop_image(image, points).shape == (5, 6, 4)
assert MineruTableOrientationClsModel._rotate_image_by_label(image, "90").shape == (10, 8, 4)
assert "cv2" not in sys.modules
""")


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_model_channel_conversion_matches_opencv(dtype: type, channels: int) -> None:
    """各种实际位深和非连续数组逐像素一致，且转换结果拥有独立连续存储。"""
    cv2 = pytest.importorskip("cv2")

    image = np.arange(8 * 10 * channels).reshape(8, 10, channels).astype(dtype)[::2, ::2]
    actual = rgb_to_bgr(image)
    np.testing.assert_array_equal(actual, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert actual.flags.c_contiguous and not np.shares_memory(actual, image)


def test_grayscale_and_alpha_conversion_keep_rounding() -> None:
    """灰度复制和非白色背景的 alpha 合成保持原来的逐通道截断顺序。"""
    cv2 = pytest.importorskip("cv2")

    image = np.random.default_rng(7).integers(0, 256, (17, 13, 4), dtype=np.uint8)
    np.testing.assert_array_equal(check_img(image[:, :, 0]), cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR))
    blue, green, red, alpha_channel = cv2.split(image)
    alpha = alpha_channel / 255
    expected = cv2.merge(
        tuple(
            (background * (1 - alpha) + channel * alpha).astype(np.uint8)
            for channel, background in ((blue, 30), (green, 20), (red, 10))
        )
    )
    np.testing.assert_array_equal(alpha_to_color(image, (10, 20, 30)), expected)


@pytest.mark.parametrize("label,rotation", [("90", 2), ("270", 0), ("0", None)])
def test_orientation_rotation_keeps_pixels_and_ownership(label: str, rotation: int | None) -> None:
    """方向评分使用的旋转保持像素、连续性和零角度副本行为。"""
    cv2 = pytest.importorskip("cv2")

    image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)[::2, ::2]
    actual = MineruTableOrientationClsModel._rotate_image_by_label(image, label)
    expected = image.copy() if rotation is None else cv2.rotate(image, rotation)
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous and not np.shares_memory(actual, image)


def test_perspective_crop_keeps_numeric_contract() -> None:
    """共享内核的倾斜裁图保留透视矩阵、三次插值和边界复制。"""
    cv2 = pytest.importorskip("cv2")

    image = np.arange(20 * 30 * 3, dtype=np.uint16).reshape(20, 30, 3)
    points = np.array([[2, 2], [21, 4], [22, 15], [1, 14]], dtype=np.float32)
    width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    expected = cv2.warpPerspective(
        image,
        cv2.getPerspectiveTransform(points, target),
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    np.testing.assert_array_equal(get_rotate_crop_image(image, points), expected)


@pytest.mark.parametrize("paddle_compatible", [False, True])
def test_formula_tensor_keeps_original_gray_and_channel_values(paddle_compatible: bool) -> None:
    """公式预处理仅替换 merge，灰度舍入和归一化计算顺序保持逐元素一致。"""
    cv2 = pytest.importorskip("cv2")

    from mineru.model.mfr.pp_formulanet.processors import UniMERNetTestTransform

    image = np.random.default_rng(11).integers(0, 256, (17, 19, 3), dtype=np.uint8)
    mean = np.array([0.7931] * 3).reshape(1, 1, 3).astype("float32")
    std = np.array([0.1738] * 3).reshape(1, 1, 3).astype("float32")
    if paddle_compatible:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        expected = (cv2.merge([gray] * 3).astype("float32") - mean * 255.0) * (1.0 / (std * 255.0))
    else:
        normalized = (image.astype("float32") * float(1 / 255.0) - mean) / std
        expected = cv2.merge([np.squeeze(cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY))] * 3)
    np.testing.assert_array_equal(UniMERNetTestTransform(paddle_compatible=paddle_compatible).transform(image), expected)


@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1), (23, 27)])
def test_formula_margin_crop_keeps_threshold_and_bounds(shape: tuple[int, int]) -> None:
    """公式裁边覆盖空白、单行单列和边界像素，保持非零框的闭区间语义。"""
    cv2 = pytest.importorskip("cv2")

    from mineru.model.mfr.pp_formulanet.processors import UniMERNetImgDecode

    processor = UniMERNetImgDecode((192, 672))
    for blank in (True, False):
        array = np.full(shape, 255, dtype=np.uint8)
        if not blank:
            array[0, 0] = 0
            array[-1, -1] = 30
        with Image.fromarray(array) as image:
            actual = processor.crop_margin(image)
            if array.max() == array.min():
                assert actual is image
                continue
            normalized = (array - array.min()) / (array.max() - array.min()) * 255
            x, y, width, height = cv2.boundingRect(cv2.findNonZero(255 * (normalized < 200).astype(np.uint8)))
            with image.crop((x, y, x + width, y + height)) as expected:
                assert actual.size == expected.size
                np.testing.assert_array_equal(np.array(actual), np.array(expected))
            actual.close()


def test_table_classifier_tensor_keeps_channel_calculation_order() -> None:
    """表格分类单张及批量预处理保持 OpenCV split/merge 的浮点计算顺序。"""
    cv2 = pytest.importorskip("cv2")

    from mineru.model.table.cls.paddle_table_cls import PaddleTableClsModel

    classifier = object.__new__(PaddleTableClsModel)
    image = np.random.default_rng(13).integers(0, 256, (79, 137, 3), dtype=np.uint8)
    scale = 256 / min(image.shape[:2])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=1)
    y, x = (resized.shape[0] - 224) // 2, (resized.shape[1] - 224) // 2
    split = list(cv2.split(resized[y : y + 224, x : x + 224]))
    for channel in range(3):
        split[channel] = split[channel].astype(np.float32)
        split[channel] *= 0.00392156862745098 / [0.229, 0.224, 0.225][channel]
        split[channel] += -[0.485, 0.456, 0.406][channel] / [0.229, 0.224, 0.225][channel]
    expected = np.stack([cv2.merge(split).transpose(2, 0, 1)]).astype(np.float32)
    np.testing.assert_array_equal(classifier.preprocess(image), expected)
    np.testing.assert_array_equal(classifier.batch_preprocess([image]), expected)


@pytest.mark.parametrize("channels", [None, 1, 3])
def test_unimer_crop_and_zero_padding_keep_shape(channels: int | None) -> None:
    """单通道和 RGB 公式图的裁边、零填充保持原形状及全部像素。"""
    cv2 = pytest.importorskip("cv2")

    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("transformers")
    from mineru.model.mfr.unimernet.unimernet_hf.unimer_swin.image_processing_unimer_swin import UnimerSwinImageProcessor

    shape = (17, 29) if channels is None else (17, 29, channels)
    image = np.full(shape, 255, dtype=np.uint8)
    image[3:15, 5:24] = 0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if channels == 3 else image.copy()
    normalized = (((gray - gray.min()) / (gray.max() - gray.min())) * 255).astype(np.uint8)
    binary = 255 * (normalized < 200).astype(np.uint8)
    x, y, width, height = cv2.boundingRect(cv2.findNonZero(binary))
    cropped = image[y : y + height, x : x + width]
    np.testing.assert_array_equal(UnimerSwinImageProcessor.crop_margin_numpy(image), cropped)
    processor = UnimerSwinImageProcessor(image_size=(32, 64))
    scale = min(32 / height, 64 / width)
    resized = cv2.resize(cropped, (int(width * scale), int(height * scale)))
    pad_x, pad_y = (64 - resized.shape[1]) // 2, (32 - resized.shape[0]) // 2
    expected = cv2.copyMakeBorder(
        resized,
        pad_y,
        32 - resized.shape[0] - pad_y,
        pad_x,
        64 - resized.shape[1] - pad_x,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    np.testing.assert_array_equal(processor.prepare_input(image), expected)


def test_unet_input_tensor_keeps_inplace_channel_exchange() -> None:
    """UNet 预处理保留原位 RGB 交换及 OpenCV 浮点减乘顺序。"""
    cv2 = pytest.importorskip("cv2")

    from mineru.model.table.rec.unet_table.table_structure_unet import TSRUnet
    from mineru.model.table.rec.unet_table.utils import resize_img

    model = object.__new__(TSRUnet)
    model.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    model.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    model.inp_height, model.inp_width = 32, 64
    image = np.random.default_rng(17).integers(0, 256, (29, 47, 3), dtype=np.uint8)
    resized, _, _ = resize_img(image, (32, 64), True)
    expected = resized.copy().astype(np.float32)
    cv2.cvtColor(expected, cv2.COLOR_BGR2RGB, expected)
    cv2.subtract(expected, np.float64(model.mean.reshape(1, -1)), expected)
    cv2.multiply(expected, 1 / np.float64(model.std.reshape(1, -1)), expected)
    actual = model.preprocess(image)
    np.testing.assert_array_equal(actual["img"], expected.transpose(2, 0, 1)[None, ...])


def test_base_dependencies_exclude_opencv_and_keep_product_extras() -> None:
    """基础声明移除 OpenCV，ONNX、llama.cpp、Torch/full 和苹果自动 extra 保持原样。"""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    from packaging.requirements import Requirement

    project = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]
    requirements = [Requirement(value) for value in project["dependencies"]]
    assert not any(requirement.name.lower().replace("_", "-").startswith("opencv") for requirement in requirements)
    names = {requirement.name for requirement in requirements}
    assert {"onnxruntime", "mineru-llama-cpp", "docvortex"} <= names
    assert "mineru[torch] ; sys_platform == 'darwin' and platform_machine == 'arm64'" in project["dependencies"]
    extras = project["optional-dependencies"]
    assert {Requirement(value).name for value in extras["torch"]} == {
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "safetensors",
    }
    assert {Requirement(value).name for value in extras["full"]} == {"mineru", "vllm", "lmdeploy", "qwen-vl-utils"}
    full_mineru = next(Requirement(value) for value in extras["full"] if Requirement(value).name == "mineru")
    assert full_mineru.extras == {"torch"}


def test_production_opencv_is_limited_to_explicit_camera_calibration() -> None:
    """静态守卫只允许旧相机标定的两个按需入口使用 cv2，不放宽整个模块。"""
    import ast

    allowed = {
        ("model/ocr/seal_det_warp.py", "CurveTextRectifier.spatial_transform"),
        ("model/ocr/seal_det_warp.py", "CurveTextRectifier.calibrate"),
    }
    found: set[tuple[str, str]] = set()
    root = _ROOT / "mineru"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            is_import = isinstance(node, ast.Import) and any(alias.name == "cv2" for alias in node.names)
            is_attribute = isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "cv2"
            if not (is_import or is_attribute):
                continue
            names = []
            parent = node
            while parent in parents:
                parent = parents[parent]
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(parent.name)
            location = (path.relative_to(root).as_posix(), ".".join(reversed(names)))
            assert location in allowed, (location, node.lineno)
            found.add(location)
    assert found == allowed


def test_model_preprocessors_and_geometry_run_without_opencv() -> None:
    """独立阻断进程执行检测后处理、识别缩放、表格线和公式图像处理。"""
    _run_blocked("""
import numpy as np
from mineru.model.ocr.db_postprocess import DBPostProcess
from mineru.model.ocr.image import get_rotate_crop_image, resize_text_recognition_image
from mineru.model.table.rec.unet_table.utils import LoadImage
from mineru.model.table.rec.unet_table.utils_table_line_rec import draw_lines, _iter_connected_component_coords
from mineru.model.mfr.pp_formulanet.processors import UniMERNetTestTransform

pred = np.zeros((1, 1, 32, 48), np.float32)
pred[0, 0, 5:12, 8:30] = .85
pred[0, 0, 19:27, 23:42] = .95
result = DBPostProcess()({"maps": pred}, np.float32([[96, 144, 1, 1]]))
assert result[0]["points"].tolist() == [[[54, 42], [138, 42], [138, 93], [54, 93]], [[9, 0], [102, 0], [102, 48], [9, 48]]]
image = np.arange(40 * 80 * 3, dtype=np.uint8).reshape(40, 80, 3)
points = np.float32([[3, 5], [74, 7], [70, 30], [1, 28]])
crop = get_rotate_crop_image(image, points)
assert crop.size and crop.dtype == np.uint8 and crop.flags.c_contiguous
assert resize_text_recognition_image(crop, 8., (3, 48, 320)).dtype == np.float32
mask = draw_lines(np.zeros((60, 90), np.uint8), [[2, 30, 87, 30]], color=255, lineW=2)
assert len(list(_iter_connected_component_coords(mask))) == 1
rgba = np.zeros((10, 20, 4), np.uint8)
assert np.all(LoadImage.cvt_four_to_three(rgba) == 255)
assert "cv2" not in sys.modules
""")

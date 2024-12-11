from abc import ABC, abstractmethod
from itertools import islice
from pathlib import Path
from typing import Any
import multiprocessing

# multiprocessing.set_start_method("fork")
from multiprocessing import Pool
import json

from loguru import logger
from tqdm import tqdm  # type: ignore

import traceback
import shutil


class OperatorBase(ABC):
    """
    `OperatorBase` 是一个抽象基类，定义了一些数据处理的基本操作。

    子类需要实现抽象方法 `get_res_item` 和 `write_res_item`，并且可以自定义生成任务列表的方法 `gen_task_arg_list`。

    Attributes:
        error_log_path (Path): 错误日志的默认存储路径。
        retry_cnt (int): 默认的重试次数。
        skip_sleep_time (int): 跳过的睡眠时间，默认值为0。
    """

    error_log_path = Path("/var/log/operator_base/operator_base.log")
    retry_cnt = 5
    skip_sleep_time = 0

    @classmethod
    @abstractmethod
    def get_res_item(cls, src_file_path: Path, retry_cnt: int) -> tuple[Any, Any]:
        """
        抽象方法，子类必须实现。

        从源文件路径中获取处理结果项。此方法应包含子类的具体数据处理逻辑。

        Args:
            src_file_path (Path): 源文件路径。
            retry_cnt (int): 重试次数。

        Returns:
            tuple[Any, Any]: 处理结果和元数据项。
        """
        pass

    @classmethod
    @abstractmethod
    def write_res_item(
        cls,
        tar_file_path: Path,
        tar_meta_file_path: Path,
        res_item: Any,
        meta_item: Any,
    ) -> None:
        """
        抽象方法，子类必须实现。

        将结果项和元数据项写入到目标文件路径和元数据文件路径中。

        Args:
            tar_file_path (Path): 结果存储文件路径。
            tar_meta_file_path (Path): 元数据存储文件路径。
            res_item (Any): 处理结果项。
            meta_item (Any): 元数据项。
        """
        tar_file_path.parent.mkdir(parents=True, exist_ok=True)
        tar_meta_file_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            open(tar_file_path, "w", encoding="utf-8") as f_w,
            open(tar_meta_file_path, "w", encoding="utf-8") as f_w_meta,
        ):
            json.dump(res_item, f_w, ensure_ascii=False)
            json.dump(
                meta_item,
                f_w_meta,
                ensure_ascii=False,
            )

    @classmethod
    def process_single_item(cls, arg) -> None:
        """
        处理单个任务项。

        对于给定的源文件路径，获取处理结果并写入目标路径。如果遇到错误，记录在日志中。

        Args:
            arg (tuple): 包含源文件路径、目标文件路径、元数据文件路径、重试次数和是否覆盖的标志。
        """
        (
            src_file_path,
            tar_file_path,
            tar_meta_file_path,
            retry_cnt,
            is_overwrite,
        ) = arg

        if not is_overwrite:
            if tar_file_path.exists():
                file_size = tar_file_path.stat().st_size
                if file_size > 512:
                    return

        try:
            res_item, meta_item = cls.get_res_item(src_file_path, retry_cnt)
        except Exception as e:
            logger.error(f"## {str(src_file_path)} ;info: {arg}; error: {str(e)}\n")
            with open(cls.error_log_path, "a", encoding="utf-8") as f_w:
                f_w.write(f"## {str(src_file_path)} ;info: {arg}; error: {str(e)}\n")
            return

        cls.write_res_item(
            tar_file_path, tar_meta_file_path, res_item=res_item, meta_item=meta_item
        )

    @abstractmethod
    def gen_task_arg_list(
        self,
        src_dir_path: Path,
        tar_dir_path: Path,
        tar_meta_dir_path: Path,
        skip_id: int,
        file_name_suffix: str,
        retry_cnt: int,
        is_overwrite: bool,
    ) -> list[tuple]:
        """
        生成任务参数列表，用于多进程处理。

        需要子类实现具体的任务生成逻辑。

        Args:
            src_dir_path (Path): 源数据目录路径。
            tar_dir_path (Path): 目标数据目录路径。
            tar_meta_dir_path (Path): 目标元数据目录路径。
            skip_id (int): 偏移值，用于跳过前skip_id个文件。
            file_name_suffix (str): 文件名后缀，用于筛选源文件。
            retry_cnt (int): 重试次数。
            is_overwrite (bool): 是否覆盖已存在的文件。

        Returns:
            list[tuple]: 含有处理任务信息的任务列表。
        """
        files_to_process = islice(
            src_dir_path.rglob(f"*{file_name_suffix}"), skip_id, None
        )
        task_arg_list = [
            (
                src_file_path,
                (tar_dir_path / src_file_path.relative_to(src_dir_path)).parent
                / (src_file_path.stem + ".json"),
                tar_meta_dir_path / ("tmp_" + src_file_path.stem + ".json"),
                retry_cnt,
                is_overwrite,
            )
            for src_file_path in files_to_process
        ]

        return task_arg_list

    def _operate_pipeline(
        self,
        src_dir_path: Path,
        tar_dir_path: Path,
        tar_meta_dir_path: Path,
        skip_id: int,
        is_debug: bool,
        multiprocess_num: int,
        retry_cnt: int,
        file_name_suffix: str,
        is_overwrite: bool,
    ) -> None:
        """
        操作处理流水线，负责组织任务并使用多进程进行处理。

        Args:
            src_dir_path (Path): 源数据目录路径。
            tar_dir_path (Path): 目标数据目录路径。
            tar_meta_dir_path (Path): 目标元数据目录路径。
            skip_id (int): 偏移值，用于跳过前skip_id个文件。
            is_debug (bool): 是否进入调试模式。
            multiprocess_num (int): 使用的多进程数量。
            retry_cnt (int): 重试次数。
            file_name_suffix (str): 文件名后缀，用于筛选源文件。
            is_overwrite (bool): 是否覆盖已存在的文件。
        """
        tar_dir_path.mkdir(parents=True, exist_ok=True)
        tar_meta_dir_path.mkdir(parents=True, exist_ok=True)
        OperatorBase.error_log_path.parent.mkdir(exist_ok=True, parents=True)

        task_arg_list = self.gen_task_arg_list(
            src_dir_path,
            tar_dir_path,
            tar_meta_dir_path,
            skip_id,
            file_name_suffix,
            retry_cnt,
            is_overwrite,
        )

        if is_debug:
            self.process_single_item(task_arg_list[-1])
            return

        if multiprocess_num == 1:
            for task_arg in tqdm(task_arg_list):
                self.process_single_item(task_arg)
        else:
            with Pool(multiprocess_num) as pool:
                list(
                    tqdm(
                        pool.imap_unordered(
                            self.__class__.process_single_item, task_arg_list
                        ),
                        desc=f"processing pipeline {src_dir_path.name}",
                    )
                )

    def __call__(
        self,
        src_dir_path: Path,
        tar_dir_path: Path,
        tar_meta_dir_path: Path | None = None,
        file_name_suffix: str = "",
        skip_id: int = 0,
        is_debug: bool = False,
        multiprocess_num: int = 1,
        retry_cnt: int = 5,
        error_log_path: Path | None = None,
        skip_sleep_time: int | None = None,
        is_overwrite: bool = False,
    ) -> None:
        """
        `__call__` 方法用于执行操作流水线。可以直接调用该实例来进行处理。

        Args:
            src_dir_path (Path): 源数据目录路径。
            tar_dir_path (Path): 目标数据目录路径。
            tar_meta_dir_path (Path): 目标元数据目录路径。
            file_name_suffix (str): 文件名后缀，用于筛选源文件。
            skip_id (int): 偏移值，用于跳过前skip_id个文件。默认0。
            is_debug (bool): 是否进入调试模式。默认False。
            multiprocess_num (int): 使用的多进程数量。默认1。
            retry_cnt (int): 重试次数。默认5。
            error_log_path (Path | None): 错误日志存储路径。默认None。
            skip_sleep_time (int | None): 跳过的睡眠时间。默认None。
            is_overwrite (bool): 是否覆盖已存在的文件。默认False。
        """
        if retry_cnt:
            OperatorBase.retry_cnt = retry_cnt
        if error_log_path:
            OperatorBase.error_log_path = error_log_path
        if skip_sleep_time:
            OperatorBase.skip_sleep_time = skip_sleep_time
        if not tar_meta_dir_path:
            tar_meta_dir_path = tar_dir_path.parent / f"meta_{tar_dir_path.name}"

        self._operate_pipeline(
            src_dir_path=src_dir_path,
            tar_dir_path=tar_dir_path,
            tar_meta_dir_path=tar_meta_dir_path,
            skip_id=skip_id,
            is_debug=is_debug,
            multiprocess_num=multiprocess_num,
            retry_cnt=retry_cnt,
            file_name_suffix=file_name_suffix,
            is_overwrite=is_overwrite,
        )

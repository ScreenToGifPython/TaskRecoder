import sys
import uuid
import datetime
import os
from openai import OpenAI
import httpx

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QListWidget, QComboBox, QMessageBox,
    QDialog, QFormLayout, QDialogButtonBox, QTableView, QFileDialog, QStatusBar,
    QCheckBox, QTextEdit, QAbstractItemView
)

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QTimer
from PySide6.QtWidgets import QDateEdit, QSizePolicy, QListWidget, QListWidgetItem
from PySide6.QtCore import QDate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
pd.set_option('display.max_columns', 1000)  # 显示字段的数量
pd.set_option('display.width', 1000)  # 表格不分段显示

# ================== 全局数据结构 ===================
# 让 task_id 仅做索引，不再保留在列中, 注意: columns里就不再包含 task_id
tasks_column_list = [
                     "title",
                     "big_category",
                     "sub_category",
                     "detail",
                     "tags",
                     "current_progress",
                     "is_finished"]
TASKS_DF = pd.DataFrame(columns=tasks_column_list)
TASKS_DF.index.name = "task_id"

# 让 log_id 仅做索引，不再保留在列中
TASK_LOGS_DF = pd.DataFrame(columns=[
    "task_id", "start_time", "end_time", "duration_min",
    "progress_final", "execution_detail", "notes"
])
TASK_LOGS_DF.index.name = "log_id"

TASKS_PARQUET_PATH = "tasks.parquet"
TASKS_PARQUET_BCK_PATH = "tasks_bck.parquet"  # 备份文件
TASK_LOGS_PARQUET_PATH = "task_logs.parquet"
TASK_LOGS_PARQUET_BCK_PATH = "task_logs_bck.parquet"  # 备份文件


# ================== 数据读取函数 ===================
def load_data():
    global TASKS_DF, TASK_LOGS_DF
    if os.path.exists(TASKS_PARQUET_PATH):
        # 读取数据
        df_tasks = pq.read_table(TASKS_PARQUET_PATH).to_pandas()
        # 判断字段是否一致
        df_tasks = df_tasks.reindex(columns=tasks_column_list)
        # 备份
        df_tasks.to_parquet(TASKS_PARQUET_BCK_PATH)
        # 设置 index
        df_tasks.set_index("task_id", inplace=True, drop=True)
        # 赋值
        TASKS_DF = df_tasks

    if os.path.exists(TASK_LOGS_PARQUET_PATH):
        # 读取数据
        df_logs = pq.read_table(TASK_LOGS_PARQUET_PATH).to_pandas()
        # 设置 index
        df_logs.set_index("log_id", inplace=True, drop=True)
        # 赋值
        TASK_LOGS_DF = df_logs.dropna()


def save_data():
    # 把 index 变回列以存储
    table_tasks = pa.Table.from_pandas(TASKS_DF.reset_index())
    # 写入
    pq.write_table(table_tasks, TASKS_PARQUET_PATH)

    # 把 index 变回列以存储
    table_logs = pa.Table.from_pandas(TASK_LOGS_DF.reset_index())
    # 写入
    pq.write_table(table_logs, TASK_LOGS_PARQUET_PATH)


def export_to_excel(filepath):
    with pd.ExcelWriter(filepath) as writer:
        TASKS_DF.reset_index().to_excel(writer, sheet_name="Tasks", index=False)
        TASK_LOGS_DF.reset_index().to_excel(writer, sheet_name="TaskLogs", index=False)


# ================== 业务逻辑函数 ===================
def create_task(title, bigcat, subcat, detail, tags):
    global TASKS_DF
    new_id = str(uuid.uuid4())[:8]
    TASKS_DF.loc[new_id] = [
        title, bigcat, subcat, detail, tags,
        0,  # current_progress
        False  # is_finished
    ]


def update_task_progress(task_id, new_progress):
    global TASKS_DF
    # 强制限制进度在 [0, 100]
    if new_progress < 0:
        new_progress = 0
    if new_progress > 100:
        new_progress = 100
    TASKS_DF.at[task_id, "current_progress"] = new_progress
    TASKS_DF.at[task_id, "is_finished"] = (new_progress >= 100)


def delete_task(task_id):
    global TASKS_DF, TASK_LOGS_DF
    if task_id in TASKS_DF.index:
        TASKS_DF.drop(index=task_id, inplace=True)
    # 同步删除其所有执行记录
    # 方式：先找出 logs 里 task_id==... 的索引
    logs_to_del = TASK_LOGS_DF[TASK_LOGS_DF["task_id"] == task_id].index
    TASK_LOGS_DF.drop(index=logs_to_del, inplace=True)


def start_task_execution(task_id):
    global TASK_LOGS_DF
    new_log_id = str(uuid.uuid4())[:8]
    # 新建一条日志
    TASK_LOGS_DF.loc[new_log_id] = [
        task_id,
        datetime.datetime.now(),  # start_time
        None,  # end_time
        0.0,  # duration_min
        None,  # progress_final
        "",  # execution_detail
        ""  # notes
    ]
    return new_log_id


def end_task_execution(log_id, final_progress, user_duration, detail, notes):
    global TASK_LOGS_DF
    end_time = datetime.datetime.now()
    TASK_LOGS_DF.at[log_id, "end_time"] = end_time

    start_time = TASK_LOGS_DF.at[log_id, "start_time"]
    if start_time is not None:
        diff_min = (end_time - start_time).total_seconds() / 60.0
    else:
        diff_min = 0
    duration_final = user_duration if user_duration > 0 else diff_min
    duration_final = round(duration_final, 2)

    # 进度也限制在 [0, 100]
    if final_progress < 0:
        final_progress = 0
    if final_progress > 100:
        final_progress = 100

    TASK_LOGS_DF.at[log_id, "duration_min"] = duration_final
    TASK_LOGS_DF.at[log_id, "progress_final"] = final_progress
    TASK_LOGS_DF.at[log_id, "execution_detail"] = detail
    TASK_LOGS_DF.at[log_id, "notes"] = notes

    # 同步更新任务进度
    task_id = TASK_LOGS_DF.at[log_id, "task_id"]
    update_task_progress(task_id, final_progress)


def delete_log(log_id):
    global TASK_LOGS_DF
    if log_id in TASK_LOGS_DF.index:
        TASK_LOGS_DF.drop(index=log_id, inplace=True)


def get_unfinished_tasks():
    return TASKS_DF[TASKS_DF["is_finished"] == False]


# ================== 合并Logs和Tasks以显示任务信息 ===================
def get_merged_logs_df(hide_finished: bool):
    """
    将 TASK_LOGS_DF 与 TASKS_DF 合并，以获得 title, big_category, sub_category 等。
    不显示 log_id / task_id，只显示:
      [title, big_category, sub_category, start_time, end_time, duration_min, progress_final, execution_detail, notes]
    若 hide_finished=True，就只保留 “对应任务 current_progress < 100” 的记录
    """
    # 先把 Logs reset_index() 以便获得 log_id 列
    logs_df = TASK_LOGS_DF.reset_index()  # columns: [log_id, task_id, start_time, ...]
    tasks_df = TASKS_DF.reset_index()  # columns: [task_id, title, big_category, sub_category, ..., is_finished]
    merged = pd.merge(logs_df, tasks_df, on="task_id", how="left")
    if hide_finished:
        merged = merged[merged["current_progress"] < 100]

    # 只保留我们要显示的列
    # 说明：log_id不再展示
    merged = merged[[
        "title", "big_category", "sub_category",
        "start_time", "end_time", "duration_min",
        "progress_final", "execution_detail", "notes"
    ]]
    return merged


def merge_and_filter_logs(start_date, end_date):
    """
    按日期范围过滤 (start_time.date() in [start_date, end_date]),
    并合并到 tasks, 只保留:
      title, big_category, sub_category,
      start_time, end_time, duration_min,
      progress_final, execution_detail
    按title排序
    """
    logs_df = TASK_LOGS_DF.reset_index()
    logs_df["date_part"] = logs_df["start_time"].apply(lambda x: x.date() if pd.notnull(x) else None)
    mask = (logs_df["date_part"] >= start_date) & (logs_df["date_part"] <= end_date)
    logs_filtered = logs_df[mask].copy()
    if logs_filtered.empty:
        return logs_filtered[[]]  # 空

    tasks_df = TASKS_DF.reset_index()  # [task_id, title, big_category, ...]
    merged = pd.merge(logs_filtered, tasks_df, on="task_id", how="left")

    # 只保留指定列, 且按title排序
    merged = merged[[
        "title", "big_category", "sub_category",
        "start_time", "end_time", "duration_min",
        "progress_final", "execution_detail"
    ]]
    merged.sort_values(by="title", inplace=True, ignore_index=True)
    return merged


def call_openai_client(api_key, base_url, model, system_prompt, user_prompt):
    """
    用 from openai import OpenAI 的客户端
    """
    try:
        print(api_key)
        print(base_url)
        print(model)
        print(system_prompt)
        print(user_prompt)

        # 直接在 Client 中传 proxies（兼容所有版本）
        http_client = httpx.Client()

        # DeepSeek 客户端
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client
        )

        # 调用
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt},
                      ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"【OpenAI调用失败】{repr(e)}"


# ============ 表格模型, 用于显示 df_filtered ============
class FilteredLogsTableModel(QAbstractTableModel):
    """
    用于在界面上显示 df_filtered:
      title, big_category, sub_category,
      start_time, end_time, duration_min,
      progress_final, execution_detail
    """

    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        if df is None:
            df = pd.DataFrame()
        self._df = df
        self.headers = [
            "任务名称", "任务大类", "任务细类",
            "开始时间", "结束时间", "耗时(分钟)",
            "进度(%)", "执行内容"
        ]

    def update_df(self, new_df):
        self.beginResetModel()
        self._df = new_df.copy()
        self.endResetModel()

    def remove_row(self, row_idx):
        """
        从 self._df 中删除某行
        """
        if 0 <= row_idx < len(self._df):
            self.beginRemoveRows(QModelIndex(), row_idx, row_idx)
            self._df.drop(self._df.index[row_idx], inplace=True)
            self.endRemoveRows()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        row = index.row()
        col = index.column()
        val = self._df.iloc[row, col]
        if pd.isnull(val):
            return ""
        if isinstance(val, datetime.datetime):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        return str(val)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.headers[section]
        return super().headerData(section, orientation, role)


# ============ 报告模板对话框 ============
class TemplateDialog(QDialog):
    """
    用于输入/修改 '报告模板'
    """

    def __init__(self, current_template="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("报告模板")
        self.template_value = current_template

        layout = QFormLayout(self)

        self.edit_template = QTextEdit()
        self.edit_template.setPlainText(current_template)
        layout.addRow(QLabel("请在下方输入你的报告模板:"))
        layout.addRow(self.edit_template)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.on_ok)
        btn_box.rejected.connect(self.reject)
        layout.addRow(btn_box)

    def on_ok(self):
        self.template_value = self.edit_template.toPlainText()
        self.accept()


# ================== 对话框：编辑/修改 任务 ===================
class EditTaskDialog(QDialog):
    def __init__(self, task_id, parent=None):
        super().__init__(parent)
        self.task_id = task_id
        self.setWindowTitle("编辑任务")
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)
        if self.task_id in TASKS_DF.index:
            old_title = TASKS_DF.at[self.task_id, "title"]
            old_bigcat = TASKS_DF.at[self.task_id, "big_category"]
            old_subcat = TASKS_DF.at[self.task_id, "sub_category"]
            old_detail = TASKS_DF.at[self.task_id, "detail"]
        else:
            old_title = ""
            old_bigcat = ""
            old_subcat = ""
            old_detail = ""

        self.edit_title = QLineEdit(str(old_title))
        self.edit_bigcat = QLineEdit(str(old_bigcat))
        self.edit_subcat = QLineEdit(str(old_subcat))
        self.edit_detail = QLineEdit(str(old_detail))

        layout.addRow("任务名称:", self.edit_title)
        layout.addRow("任务大类:", self.edit_bigcat)
        layout.addRow("任务细类:", self.edit_subcat)
        layout.addRow("任务明细:", self.edit_detail)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btn_box.accepted.connect(self.on_ok)
        btn_box.rejected.connect(self.reject)

        layout.addRow(btn_box)

    def on_ok(self):
        new_title = self.edit_title.text().strip()
        new_bigcat = self.edit_bigcat.text().strip()
        new_subcat = self.edit_subcat.text().strip()
        new_detail = self.edit_detail.text().strip()

        if not new_title or not new_bigcat or not new_subcat:
            QMessageBox.warning(self, "警告", "任务名称、大类、细类 不能为空！")
            return

        if self.task_id in TASKS_DF.index:
            TASKS_DF.at[self.task_id, "title"] = new_title
            TASKS_DF.at[self.task_id, "big_category"] = new_bigcat
            TASKS_DF.at[self.task_id, "sub_category"] = new_subcat
            TASKS_DF.at[self.task_id, "detail"] = new_detail

            save_data()
            QMessageBox.information(self, "提示", "已更新任务信息")
            self.accept()
        else:
            QMessageBox.warning(self, "警告", "task_id 不存在，无法编辑！")


# ================== 对话框：编辑/修改 执行记录 ===================
class EditLogDialog(QDialog):
    """
    用于编辑执行记录的 耗时、进度、执行内容、备注
    """

    def __init__(self, log_id, parent=None):
        super().__init__(parent)
        self.log_id = log_id
        self.setWindowTitle("编辑执行记录")
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)
        if self.log_id in TASK_LOGS_DF.index:
            old_dur = TASK_LOGS_DF.at[self.log_id, "duration_min"]
            old_prog = TASK_LOGS_DF.at[self.log_id, "progress_final"]
            old_detail = TASK_LOGS_DF.at[self.log_id, "execution_detail"]
            old_notes = TASK_LOGS_DF.at[self.log_id, "notes"]
        else:
            old_dur = 0.0
            old_prog = 0.0
            old_detail = ""
            old_notes = ""

        if old_prog is None:
            old_prog = 0

        self.edit_duration = QLineEdit(str(old_dur if old_dur else 0.0))
        self.edit_progress = QLineEdit(str(old_prog))
        self.edit_detail = QLineEdit(str(old_detail if old_detail else ""))
        self.edit_notes = QLineEdit(str(old_notes if old_notes else ""))

        layout.addRow("耗时(分钟):", self.edit_duration)
        layout.addRow("进度(0-100):", self.edit_progress)
        layout.addRow("执行内容:", self.edit_detail)
        layout.addRow("备注:", self.edit_notes)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btn_box.accepted.connect(self.on_ok)
        btn_box.rejected.connect(self.reject)

        layout.addRow(btn_box)

    def on_ok(self):
        if self.log_id not in TASK_LOGS_DF.index:
            QMessageBox.warning(self, "警告", "log_id 不存在，无法编辑！")
            return

        try:
            new_dur = float(self.edit_duration.text().strip())
        except:
            QMessageBox.warning(self, "警告", "耗时请输入数字！")
            return

        try:
            new_prog = float(self.edit_progress.text().strip())
        except:
            QMessageBox.warning(self, "警告", "进度请输入数字！")
            return

        # 限制进度在 [0, 100]
        if new_prog < 0:
            new_prog = 0
        if new_prog > 100:
            new_prog = 100

        new_detail = self.edit_detail.text().strip()
        new_notes = self.edit_notes.text().strip()

        TASK_LOGS_DF.at[self.log_id, "duration_min"] = new_dur
        TASK_LOGS_DF.at[self.log_id, "progress_final"] = new_prog
        TASK_LOGS_DF.at[self.log_id, "execution_detail"] = new_detail
        TASK_LOGS_DF.at[self.log_id, "notes"] = new_notes

        # 同步更新任务进度
        task_id = TASK_LOGS_DF.at[self.log_id, "task_id"]
        update_task_progress(task_id, new_prog)

        save_data()
        QMessageBox.information(self, "提示", "已更新执行记录")
        self.accept()


# ================== 结束执行时的弹窗对话框 ===================
class EndExecutionDialog(QDialog):
    def __init__(self, log_id, parent=None):
        super().__init__(parent)
        self.log_id = log_id
        self.setWindowTitle("结束执行 - 填写执行明细")
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)

        if self.log_id not in TASK_LOGS_DF.index:
            QMessageBox.warning(self, "错误", "log_id 不存在，无法结束执行")
            self.close()
            return

        start_time = TASK_LOGS_DF.at[self.log_id, "start_time"]
        if start_time is None:
            start_time = datetime.datetime.now()
        diff_min = round((datetime.datetime.now() - start_time).total_seconds() / 60.0, 2)

        task_id = TASK_LOGS_DF.at[self.log_id, "task_id"]
        if task_id in TASKS_DF.index:
            current_prog = TASKS_DF.at[task_id, "current_progress"]
        else:
            current_prog = 0

        self.edit_progress = QLineEdit(str(current_prog if current_prog else 0))
        self.edit_duration = QLineEdit(str(diff_min))
        self.edit_detail = QLineEdit()
        self.edit_notes = QLineEdit()

        layout.addRow("最新进度(0-100):", self.edit_progress)
        layout.addRow("本次执行耗时(分钟):", self.edit_duration)
        layout.addRow("执行内容:", self.edit_detail)
        layout.addRow("备注:", self.edit_notes)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btn_box.accepted.connect(self.on_ok)
        btn_box.rejected.connect(self.reject)
        layout.addRow(btn_box)

    def on_ok(self):
        if self.log_id not in TASK_LOGS_DF.index:
            QMessageBox.warning(self, "警告", "当前 log_id 不存在，无法结束执行")
            return

        try:
            final_progress = float(self.edit_progress.text().strip())
        except:
            QMessageBox.warning(self, "警告", "进度请输入数字！")
            return

        try:
            user_duration = float(self.edit_duration.text().strip())
        except:
            user_duration = 0.0

        # 限制进度在 [0, 100]
        if final_progress < 0:
            final_progress = 0
        if final_progress > 100:
            final_progress = 100

        detail = self.edit_detail.text().strip()
        notes = self.edit_notes.text().strip()

        end_task_execution(self.log_id, final_progress, user_duration, detail, notes)
        save_data()
        # QMessageBox.information(self, "提示", "执行信息已保存")
        self.accept()


# ================== 日志表的 TableModel ===================
class LogsTableModel(QAbstractTableModel):
    """
    不再显示 log_id 和 task_id，而是显示:
    任务名称( title ), 任务大类( big_category ), 任务细类( sub_category ),
    开始时间, 结束时间, 耗时(分钟), 进度, 执行内容, 备注
    可通过 hide_finished 来过滤已完成的任务
    """

    def __init__(self, hide_finished=False, parent=None):
        super().__init__(parent)
        self.hide_finished = hide_finished
        self._df = get_merged_logs_df(self.hide_finished)
        self.headers = [
            "任务名称", "任务大类", "任务细类",
            "开始时间", "结束时间", "耗时(分钟)",
            "进度(%)", "执行内容", "备注"
        ]

    def set_hide_finished(self, hide: bool):
        self.hide_finished = hide
        self.update_data()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        row = index.row()
        col = index.column()
        val = self._df.iloc[row, col]

        if isinstance(val, datetime.datetime):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        return str(val) if pd.notnull(val) else ""

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.headers[section]
        return super().headerData(section, orientation, role)

    def update_data(self):
        self.beginResetModel()
        self._df = get_merged_logs_df(self.hide_finished)
        self.endResetModel()

    def get_row_data(self, row_index):
        if row_index < 0 or row_index >= len(self._df):
            return None
        return self._df.iloc[row_index].to_dict()


class StatsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        date_layout = QHBoxLayout()
        self.start_date = QDateEdit()
        self.end_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addDays(-7))
        self.end_date.setDate(QDate.currentDate())
        self.start_date.setCalendarPopup(True)
        self.end_date.setCalendarPopup(True)
        date_layout.addWidget(QLabel("开始日期:"))
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(QLabel("结束日期:"))
        date_layout.addWidget(self.end_date)

        # 初始化时：
        self.category_map = {
            "任务名称": "title",
            "任务大类": "big_category",
            "任务细类": "sub_category"
        }

        self.category_selector = QComboBox()
        self.category_selector.addItems(self.category_map.keys())
        self.category_selector.currentIndexChanged.connect(self.update_filter_options)

        layout.addLayout(date_layout)

        filter_layout = QVBoxLayout()
        filter_layout.addWidget(QLabel("选择要展示的分类项（可多选）:"))
        self.filter_list = QListWidget()
        self.filter_list.setSelectionMode(QListWidget.MultiSelection)
        self.filter_list.setMaximumHeight(100)
        filter_layout.addWidget(self.filter_list)
        layout.addLayout(filter_layout)

        # self.category_selector.addItems(["title", "big_category", "sub_category"])
        # self.category_selector.addItems(["任务名称", "任务大类", "任务细类"])
        date_layout.addWidget(QLabel("统计维度:"))
        date_layout.addWidget(self.category_selector)

        self.query_button = QPushButton("生成统计图")
        self.query_button.clicked.connect(self.plot_pie_chart)
        date_layout.addWidget(self.query_button)

        self.export_button = QPushButton("导出图表为图片")
        self.export_button.clicked.connect(self.export_chart)
        date_layout.addWidget(self.export_button)

        layout.addLayout(date_layout)

        # self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # self.canvas = FigureCanvas(self.figure)
        # self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # layout.addWidget(self.canvas)
        self.canvas = None  # 暂时不显示图表

        self.setLayout(layout)

    def update_filter_options(self):
        display_text = self.category_selector.currentText()
        group_col = self.category_map[display_text]
        merged = TASK_LOGS_DF.merge(TASKS_DF, left_on="task_id", right_index=True, how="left")
        unique_values = merged[group_col].dropna().unique().tolist()

        self.filter_list.clear()
        for item in sorted(unique_values):
            list_item = QListWidgetItem(str(item))
            list_item.setSelected(True)
            self.filter_list.addItem(list_item)

    def plot_pie_chart(self):
        if self.canvas is None:
            self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.layout().addWidget(self.canvas)
        self.ax1.clear()
        self.ax2.clear()
        start = pd.Timestamp(self.start_date.date().toPython())
        end = pd.Timestamp(self.end_date.date().toPython()) + pd.Timedelta(days=1)

        mask = (TASK_LOGS_DF['start_time'] >= start) & (TASK_LOGS_DF['start_time'] < end)
        df_filtered = TASK_LOGS_DF.loc[mask]

        if df_filtered.empty:
            QMessageBox.information(self, "无数据", "该时间段内没有任务记录")
            self.canvas.draw()
            return

        merged = df_filtered.merge(TASKS_DF, left_on="task_id", right_index=True, how="left")
        # group_col = self.category_selector.currentText()
        display_text = self.category_selector.currentText()
        group_col = self.category_map[display_text]
        selected_items = [item.text() for item in self.filter_list.selectedItems()]
        if selected_items:
            merged = merged[merged[group_col].isin(selected_items)]

        group = merged.groupby(group_col)["duration_min"].sum().sort_values(ascending=False)

        group.plot.pie(autopct='%1.1f%%', ax=self.ax1, ylabel="")
        self.ax1.set_title(f"按 {self.category_selector.currentText()} 的饼图")

        group.plot.bar(ax=self.ax2)
        self.ax2.set_ylabel("总时长（分钟）")
        self.ax2.set_title(f"按 {self.category_selector.currentText()} 的柱状图")
        self.ax2.tick_params(axis='x', rotation=45)

        self.figure.tight_layout()
        self.canvas.draw()

    def export_chart(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "保存图表", "chart.png", "PNG 文件 (*.png)")
        if filepath:
            self.figure.savefig(filepath)
            QMessageBox.information(self, "导出成功", f"图表已保存到：{filepath}")


# ================== 主窗口 MainWindow ===================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NewBeePlus 任务管理系统 V2.0")

        load_data()

        self.report_template = ""  # 保存用户自定义的报告模板

        self.current_log_id = None
        self.start_time_for_timer = None  # 用于实时显示执行耗时

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Tab1: 任务管理
        self.tab_tasks = QWidget()
        self.tab_widget.addTab(self.tab_tasks, "任务管理")
        self.init_tab_tasks()

        # Tab2: 执行管理
        self.tab_execute = QWidget()
        self.tab_widget.addTab(self.tab_execute, "执行管理")
        self.init_tab_execute()

        # Tab3: 执行明细
        self.tab_logs = QWidget()
        self.tab_widget.addTab(self.tab_logs, "执行明细")
        self.tab_stats = StatsTab()
        self.tab_widget.addTab(self.tab_stats, "工作统计")
        self.init_tab_logs()

        # Tab4:工作统计
        self.setStatusBar(QStatusBar())
        export_btn = QPushButton("导出Excel")
        export_btn.clicked.connect(self.on_export_excel)
        self.statusBar().addPermanentWidget(export_btn)

        # Tab5: 生成简报
        self.tab_summary = QWidget()
        self.tab_widget.addTab(self.tab_summary, "生成简报")
        self.init_tab_summary()

        self.resize(1000, 600)

        # 最后刷新
        self.refresh_task_lists()
        self.refresh_logs_table()

    '''========== Tab1: 任务管理 ============='''

    # 初始化任务管理界面
    def init_tab_tasks(self):
        layout = QVBoxLayout(self.tab_tasks)

        self.check_hide_finished_tasks = QCheckBox("隐藏已完成的任务")
        self.check_hide_finished_tasks.stateChanged.connect(self.on_hide_finished_tasks_changed)
        layout.addWidget(self.check_hide_finished_tasks, alignment=Qt.AlignLeft)

        hbox = QHBoxLayout()
        layout.addLayout(hbox)

        # 左侧：创建任务
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.edit_title = QLineEdit()
        self.edit_title.setPlaceholderText("请输入任务的名称")  # 设置占位符文本

        self.edit_bigcat = QComboBox()
        self.edit_bigcat.setEditable(True)
        # bug_list = list(set(['其他'] + TASKS_DF['big_category'].tolist()))
        bug_list = ['个人', '公司', '其他']
        self.edit_bigcat.addItems(bug_list)  # 可按需初始化

        self.edit_subcat = QComboBox()
        self.edit_subcat.setEditable(True)
        sub_list = list(set(['其他'] + TASKS_DF['sub_category'].tolist()))
        self.edit_subcat.addItems(sub_list)  # 可按需初始化

        self.edit_detail = QLineEdit()
        self.edit_detail.setPlaceholderText("请输入任务详情")  # 设置占位符文本

        self.edit_tags = QLineEdit()
        self.edit_tags.setPlaceholderText("请输入任务标签,多个标签用逗号隔开")  # 设置占位符文本

        left_layout.addWidget(QLabel("任务名称:"))
        left_layout.addWidget(self.edit_title)
        left_layout.addWidget(QLabel("任务大类:"))
        left_layout.addWidget(self.edit_bigcat)
        left_layout.addWidget(QLabel("任务细类:"))
        left_layout.addWidget(self.edit_subcat)
        left_layout.addWidget(QLabel("任务明细:"))
        left_layout.addWidget(self.edit_detail)
        left_layout.addWidget(QLabel("任务标签:"))
        left_layout.addWidget(self.edit_tags)

        btn_create = QPushButton("创建任务")
        btn_create.clicked.connect(self.on_create_task)
        left_layout.addWidget(btn_create)

        hbox.addWidget(left_widget)

        # 右侧：列表 + 编辑/删除
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        right_layout.addWidget(QLabel("所有任务列表（只显示名称与进度）:"))
        self.list_tasks = QListWidget()
        right_layout.addWidget(self.list_tasks, 1)

        # 编辑 & 删除按钮
        btns_layout = QHBoxLayout()
        self.btn_edit_task = QPushButton("编辑选中任务")
        self.btn_edit_task.clicked.connect(self.on_edit_task)
        btns_layout.addWidget(self.btn_edit_task)

        self.btn_del_task = QPushButton("删除选中任务")
        self.btn_del_task.clicked.connect(self.on_del_task)
        btns_layout.addWidget(self.btn_del_task)

        right_layout.addLayout(btns_layout)

        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self.refresh_task_lists)
        right_layout.addWidget(btn_refresh)

        hbox.addWidget(right_widget)

    # 隐藏已完成任务的复选框
    def on_hide_finished_tasks_changed(self, state):
        self.refresh_task_lists()

    # 创建任务
    def on_create_task(self):
        # 获取输入: 任务名称
        title = self.edit_title.text().strip()
        # 获取输入: 大类、细类
        bigcat = self.edit_bigcat.currentText().strip()
        subcat = self.edit_subcat.currentText().strip()
        # 更新下拉框: 大类、细类
        if bigcat and (self.edit_bigcat.findText(bigcat) == -1):
            self.edit_bigcat.addItem(bigcat)
        if subcat and (self.edit_subcat.findText(subcat) == -1):
            self.edit_subcat.addItem(subcat)
        # 获取输入: 任务明细、标签
        detail = self.edit_detail.text().strip()
        tags = self.edit_tags.text().strip()

        # 检查输入
        if not title or not bigcat or not subcat:
            QMessageBox.warning(self, "警告", "任务名称、大类、细类不能为空!")
            return

        # 创建任务
        create_task(title, bigcat, subcat, detail, tags)
        # 保存数据
        save_data()
        # 提示
        QMessageBox.information(self, "提示", "创建任务成功!")

        self.refresh_task_lists()
        self.refresh_logs_table()

        self.edit_title.clear()
        # self.edit_bigcat.clear()
        # self.edit_subcat.clear()
        self.edit_detail.clear()
        self.edit_tags.clear()

    # 刷新任务列表
    def refresh_task_lists(self):
        self.list_tasks.clear()

        hide_finished = (self.check_hide_finished_tasks.isChecked())
        df_all = TASKS_DF.copy()
        if hide_finished:
            df_all = df_all[df_all["is_finished"] == False]

        for task_id, row in df_all.iterrows():
            # 只显示 任务名称 + 进度
            line = f"{row.title} (进度 {row.current_progress}%)"
            self.list_tasks.addItem(line)

        # 同步更新“执行管理”下拉(显示未完成任务)
        if hasattr(self, "combo_unfinished"):
            self.combo_unfinished.clear()
            df_unfinish = get_unfinished_tasks()
            for t_id, r in df_unfinish.iterrows():
                disp = f"{t_id} | {r.title}(进度{r.current_progress}%)"
                self.combo_unfinished.addItem(disp)

    # 编辑任务
    def on_edit_task(self):
        selected_items = self.list_tasks.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选中一个任务！")
            return

        selected_text = selected_items[0].text()  # e.g. "XXX (进度 0%)"
        # 反查 task_id
        found_id = None
        for t_id, row in TASKS_DF.iterrows():
            text_line = f"{row.title} (进度 {row.current_progress}%)"
            if text_line == selected_text:
                found_id = t_id
                break
        if not found_id:
            QMessageBox.warning(self, "警告", "未找到对应task_id！")
            return

        dlg = EditTaskDialog(found_id, self)
        if dlg.exec() == QDialog.Accepted:
            self.refresh_task_lists()
            self.refresh_logs_table()

    # 删除任务
    def on_del_task(self):
        selected_items = self.list_tasks.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选中一个任务！")
            return

        selected_text = selected_items[0].text()
        found_id = None
        for t_id, row in TASKS_DF.iterrows():
            text_line = f"{row.title} (进度 {row.current_progress}%)"
            if text_line == selected_text:
                found_id = t_id
                break
        if not found_id:
            QMessageBox.warning(self, "警告", "未找到对应task_id！")
            return

        ret = QMessageBox.question(self, "确认", f"确定要删除任务 {found_id} 及其执行记录？",
                                   QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            delete_task(found_id)
            save_data()
            QMessageBox.information(self, "提示", "任务及其执行记录已删除。")
            self.refresh_task_lists()
            self.refresh_logs_table()

    '''========== Tab2: 执行管理 ==========='''

    # 初始化执行管理界面
    def init_tab_execute(self):
        layout = QVBoxLayout(self.tab_execute)

        top_hbox = QHBoxLayout()
        top_hbox.addWidget(QLabel("选择未完成任务:"))
        self.combo_unfinished = QComboBox()
        top_hbox.addWidget(self.combo_unfinished)
        layout.addLayout(top_hbox)

        bottom_hbox = QHBoxLayout()

        self.btn_start = QPushButton("开始执行")
        self.btn_start.clicked.connect(self.on_start_execution)
        bottom_hbox.addWidget(self.btn_start)

        self.btn_end = QPushButton("结束执行")
        self.btn_end.clicked.connect(self.on_end_execution)
        self.btn_end.setEnabled(False)
        bottom_hbox.addWidget(self.btn_end)

        self.lbl_elapsed_time = QLabel("耗时: 00:00:00")
        bottom_hbox.addWidget(self.lbl_elapsed_time)

        self.lbl_current_log = QLabel("当前执行log_id: 无")
        bottom_hbox.addWidget(self.lbl_current_log)

        layout.addLayout(bottom_hbox)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_elapsed_time)

    def on_start_execution(self):
        text = self.combo_unfinished.currentText().strip()
        if not text:
            QMessageBox.warning(self, "警告", "请先选择一个未完成的任务!")
            return

        # 解析 "task_id | title(进度xx%)"
        parts = text.split("|", maxsplit=1)
        if len(parts) < 2:
            QMessageBox.warning(self, "警告", "无法解析task_id")
            return
        task_id = parts[0].strip()

        if task_id not in TASKS_DF.index:
            QMessageBox.warning(self, "警告", f"task_id={task_id} 不存在！")
            return

        self.current_log_id = start_task_execution(task_id)
        save_data()

        self.lbl_current_log.setText(f"当前执行log_id: {self.current_log_id}")
        self.btn_end.setEnabled(True)
        self.btn_start.setEnabled(False)

        # 开始计时
        self.start_time_for_timer = datetime.datetime.now()
        self.timer.start(1000)

        # QMessageBox.information(self, "提示", f"已开始执行任务 {task_id} (log_id={self.current_log_id})")

    def on_end_execution(self):
        if not self.current_log_id:
            QMessageBox.warning(self, "警告", "当前没有进行中的log_id!")
            return

        self.timer.stop()

        dlg = EndExecutionDialog(self.current_log_id, self)
        if dlg.exec() == QDialog.Accepted:
            self.current_log_id = None
            self.lbl_current_log.setText("当前执行log_id: 无")
            self.btn_end.setEnabled(False)
            self.btn_start.setEnabled(True)
            self.lbl_elapsed_time.setText("耗时: 00:00:00")

            self.refresh_task_lists()
            self.refresh_logs_table()

    def update_elapsed_time(self):
        if not self.start_time_for_timer:
            return
        delta = datetime.datetime.now() - self.start_time_for_timer
        total_seconds = int(delta.total_seconds())
        hh = total_seconds // 3600
        mm = (total_seconds % 3600) // 60
        ss = total_seconds % 60
        self.lbl_elapsed_time.setText(f"耗时: {hh:02d}:{mm:02d}:{ss:02d}")

    # ========== Tab3: 执行明细 ===========
    def init_tab_logs(self):
        layout = QVBoxLayout(self.tab_logs)

        self.check_hide_finished_logs = QCheckBox("隐藏已完成的任务")
        self.check_hide_finished_logs.stateChanged.connect(self.on_hide_finished_logs_changed)
        layout.addWidget(self.check_hide_finished_logs)

        self.logs_view = QTableView()
        self.logs_model = LogsTableModel(hide_finished=False)
        self.logs_view.setModel(self.logs_model)
        layout.addWidget(self.logs_view, 1)

        btns_hbox = QHBoxLayout()
        self.btn_edit_log = QPushButton("编辑选中记录")
        self.btn_edit_log.clicked.connect(self.on_edit_log)
        btns_hbox.addWidget(self.btn_edit_log)

        self.btn_del_log = QPushButton("删除选中记录")
        self.btn_del_log.clicked.connect(self.on_del_log)
        btns_hbox.addWidget(self.btn_del_log)

        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self.refresh_logs_table)
        btns_hbox.addWidget(btn_refresh)

        layout.addLayout(btns_hbox)

    def on_hide_finished_logs_changed(self, state):
        hide = True if state == 2 else False
        self.logs_model.set_hide_finished(hide)
        self.refresh_logs_table()

    def refresh_logs_table(self):
        self.logs_model.update_data()
        self.logs_view.resizeColumnsToContents()

    def on_edit_log(self):
        # 获取当前选中行
        index = self.logs_view.currentIndex()
        if not index.isValid():
            QMessageBox.warning(self, "警告", "请先选中一条执行记录！")
            return
        row = index.row()
        # get the row data
        row_data = self.logs_model._df.iloc[row]  # type: pd.Series

        # 要反查 log_id:
        # logs_model._df 里含有 "start_time", "end_time", ... 但不含 "log_id"
        # 我们可以 reset_index() 后 match, 或者(更简单) 直接保留 "log_id" in the model
        # 这里演示 reset_index:
        logs_reset = TASK_LOGS_DF.reset_index()  # columns: [log_id, task_id, start_time, ...]
        # matches
        st = row_data["start_time"]
        ed = row_data["end_time"]
        dt = row_data["execution_detail"]
        nt = row_data["notes"]
        dur = row_data["duration_min"] if "duration_min" in row_data else None
        prg = row_data["progress_final"] if "progress_final" in row_data else None

        cand = logs_reset[
            (logs_reset["start_time"] == st) &
            (logs_reset["end_time"] == ed) &
            (logs_reset["execution_detail"] == dt) &
            (logs_reset["notes"] == nt) &
            (logs_reset["duration_min"] == dur) &
            (logs_reset["progress_final"] == prg)
            ]
        if len(cand) == 0:
            QMessageBox.warning(self, "警告", "无法找到匹配的log_id！可能有重复记录。")
            return
        real_log_id = cand.iloc[0]["log_id"]

        dlg = EditLogDialog(real_log_id, self)
        if dlg.exec() == QDialog.Accepted:
            self.refresh_logs_table()
            self.refresh_task_lists()

    def on_del_log(self):
        index = self.logs_view.currentIndex()
        if not index.isValid():
            QMessageBox.warning(self, "警告", "请先选中一条执行记录！")
            return
        row = index.row()
        row_data = self.logs_model._df.iloc[row]

        st = row_data["start_time"]
        ed = row_data["end_time"]
        dt = row_data["execution_detail"]
        nt = row_data["notes"]
        dur = row_data["duration_min"] if "duration_min" in row_data else None
        prg = row_data["progress_final"] if "progress_final" in row_data else None

        logs_reset = TASK_LOGS_DF.reset_index()
        cand = logs_reset[
            (logs_reset["start_time"] == st) &
            (logs_reset["end_time"] == ed) &
            (logs_reset["execution_detail"] == dt) &
            (logs_reset["notes"] == nt) &
            (logs_reset["duration_min"] == dur) &
            (logs_reset["progress_final"] == prg)
            ]
        if len(cand) == 0:
            QMessageBox.warning(self, "警告", "无法找到匹配的log_id！可能有重复记录。")
            return
        real_log_id = cand.iloc[0]["log_id"]

        ret = QMessageBox.question(self, "确认", f"确定删除此记录？log_id={real_log_id}",
                                   QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            delete_log(real_log_id)
            save_data()
            QMessageBox.information(self, "提示", "执行记录已删除")
            self.refresh_logs_table()

    # ========== 导出Excel ==========
    def on_export_excel(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出Excel", "",
            "Excel 文件 (*.xlsx);;所有文件 (*)"
        )
        if filepath:
            export_to_excel(filepath)
            QMessageBox.information(self, "提示", f"已导出到: {filepath}")

    def init_tab_summary(self):
        layout = QVBoxLayout(self.tab_summary)

        # 1) 行1: API Key, Base URL, Model
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("API Key:"))
        self.edit_api_key = QLineEdit()
        config_layout.addWidget(self.edit_api_key)

        config_layout.addWidget(QLabel("Base URL:"))
        self.edit_base_url = QLineEdit("https://api.openai.com/v1")
        config_layout.addWidget(self.edit_base_url)

        config_layout.addWidget(QLabel("Model:"))
        self.edit_model = QLineEdit("gpt-3.5-turbo")
        config_layout.addWidget(self.edit_model)

        layout.addLayout(config_layout)

        # 2) 行2: 时间范围 + 快速日期按钮
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("开始日期:"))
        self.date_start = QDateEdit()
        self.date_start.setCalendarPopup(True)
        date_layout.addWidget(self.date_start)

        date_layout.addWidget(QLabel("结束日期:"))
        self.date_end = QDateEdit()
        self.date_end.setCalendarPopup(True)
        date_layout.addWidget(self.date_end)

        btn_today = QPushButton("今日")
        btn_today.clicked.connect(self.on_pick_today)
        date_layout.addWidget(btn_today)

        btn_yesterday = QPushButton("昨日")
        btn_yesterday.clicked.connect(self.on_pick_yesterday)
        date_layout.addWidget(btn_yesterday)

        btn_thisweek = QPushButton("本周以来")
        btn_thisweek.clicked.connect(self.on_pick_thisweek)
        date_layout.addWidget(btn_thisweek)

        btn_filter = QPushButton("筛选日志")
        btn_filter.clicked.connect(self.on_filter_logs)
        date_layout.addWidget(btn_filter)

        layout.addLayout(date_layout)

        # 3) 表格区: 显示 df_filtered
        self.table_view = QTableView()
        self.table_model = FilteredLogsTableModel()
        self.table_view.setModel(self.table_model)
        # 允许选中行
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.table_view, 1)

        # 4) 操作按钮区
        op_layout = QHBoxLayout()

        # 报告模板按钮
        self.btn_template = QPushButton("报告模板")
        self.btn_template.clicked.connect(self.on_set_report_template)
        op_layout.addWidget(self.btn_template)

        # 剔除行按钮
        self.btn_remove_row = QPushButton("剔除所选行")
        self.btn_remove_row.clicked.connect(self.on_remove_selected_row)
        op_layout.addWidget(self.btn_remove_row)

        # 下载明细
        self.btn_download = QPushButton("下载明细")
        self.btn_download.clicked.connect(self.on_download_df)
        op_layout.addWidget(self.btn_download)

        # 生成简报
        self.btn_summary = QPushButton("生成中文简报")
        self.btn_summary.clicked.connect(self.on_generate_summary)
        op_layout.addWidget(self.btn_summary)

        layout.addLayout(op_layout)

        # 5) 输出简报的文本框
        self.text_summary = QTextEdit()
        layout.addWidget(self.text_summary, 1)

    # ============ 快速日期按钮 =============
    def on_pick_today(self):
        today = datetime.date.today()
        self.date_start.setDate(QDate(today.year, today.month, today.day))
        self.date_end.setDate(QDate(today.year, today.month, today.day))

    def on_pick_yesterday(self):
        yest = datetime.date.today() - datetime.timedelta(days=1)
        self.date_start.setDate(QDate(yest.year, yest.month, yest.day))
        self.date_end.setDate(QDate(yest.year, yest.month, yest.day))

    def on_pick_thisweek(self):
        today = datetime.date.today()
        weekday = today.weekday()  # Monday=0
        monday = today - datetime.timedelta(days=weekday)
        self.date_start.setDate(QDate(monday.year, monday.month, monday.day))
        self.date_end.setDate(QDate(today.year, today.month, today.day))

    # ============ 筛选日志 & 更新表格 =============
    def on_filter_logs(self):
        start_qdate = self.date_start.date()
        end_qdate = self.date_end.date()
        start_date = datetime.date(start_qdate.year(), start_qdate.month(), start_qdate.day())
        end_date = datetime.date(end_qdate.year(), end_qdate.month(), end_qdate.day())

        if end_date < start_date:
            QMessageBox.warning(self, "警告", "结束日期不能早于开始日期！")
            return

        df_filtered = merge_and_filter_logs(start_date, end_date)
        self.table_model.update_df(df_filtered if df_filtered is not None else pd.DataFrame())

        if df_filtered is not None and not df_filtered.empty:
            # QMessageBox.information(self, "提示", f"共筛选出 {len(df_filtered)} 条记录.")
            pass
        else:
            QMessageBox.information(self, "提示", "没有查询到任何记录.")

    # ============ 报告模板 ============
    def on_set_report_template(self):
        dlg = TemplateDialog(current_template=self.report_template, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.report_template = dlg.template_value
            QMessageBox.information(self, "提示", "已更新报告模板.")

    # ============ 剔除所选行 ============
    def on_remove_selected_row(self):
        index = self.table_view.currentIndex()
        if not index.isValid():
            QMessageBox.warning(self, "警告", "请先选中一行！")
            return
        row = index.row()
        self.table_model.remove_row(row)
        # QMessageBox.information(self, "提示", f"已剔除第 {row} 行.")

    # ============ 下载明细 ============
    def on_download_df(self):
        df_current = self.table_model._df
        if df_current.empty:
            QMessageBox.information(self, "提示", "当前表格为空, 无法下载.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "另存为", "",
                                              "Excel 文件(*.xlsx);;CSV 文件(*.csv)")
        if not path:
            return

        try:
            if path.endswith(".csv"):
                df_current.to_csv(path, index=False, encoding="utf-8-sig")
            else:
                df_current.to_excel(path, index=False)
            QMessageBox.information(self, "提示", f"已保存到: {path}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存失败: {repr(e)}")

    # ============ 生成中文简报 ============
    def on_generate_summary(self):
        api_key = self.edit_api_key.text().strip()
        base_url = self.edit_base_url.text().strip()
        model = self.edit_model.text().strip()
        if not api_key or not base_url or not model:
            QMessageBox.warning(self, "警告", "请先填写 API Key, Base URL, 和 model!")
            return

        df_current = self.table_model._df
        if df_current.empty:
            QMessageBox.information(self, "提示", "当前没有任何数据, 无法生成简报.")
            return

        # 拼接 prompt
        system_prompt = "You are a helpful assistant, respond in Chinese."
        user_prompt = "以下是指定时间范围内的执行记录信息:\n"
        for i, row in df_current.iterrows():
            title = row["title"]
            bigcat = row["big_category"]
            subcat = row["sub_category"]
            st = row["start_time"]
            ed = row["end_time"]
            dur = row["duration_min"]
            prog = row["progress_final"]
            exe = row["execution_detail"]
            user_prompt += f"- 任务:{title}, 大类:{bigcat}, 细类:{subcat}, 开始:{st}, 结束:{ed}, 耗时:{dur}, 进度:{prog}, 内容:{exe}\n"

        # 如果用户填写了自定义模板, 也贴到 user_prompt 后面
        if self.report_template.strip():
            user_prompt += f"\n[用户自定义模板]:\n{self.report_template}\n"

        user_prompt += "\n请用简短中文概括本段时间内的执行情况."

        # 调用OpenAI
        result = call_openai_client(api_key, base_url, model, system_prompt, user_prompt)
        self.text_summary.setPlainText(result)


def main():
    print('''
                              8888888 8888888888   .8.            d888888o.   8 8888     ,88'                         
                                    8 8888        .888.         .`8888:' `88. 8 8888    ,88'                          
                                    8 8888       :88888.        8.`8888.   Y8 8 8888   ,88'                           
                                    8 8888      . `88888.       `8.`8888.     8 8888  ,88'                            
                                    8 8888     .8. `88888.       `8.`8888.    8 8888 ,88'                             
                                    8 8888    .8`8. `88888.       `8.`8888.   8 8888 88'                              
                                    8 8888   .8' `8. `88888.       `8.`8888.  8 888888<                               
                                    8 8888  .8'   `8. `88888.  8b   `8.`8888. 8 8888 `Y8.                             
                                    8 8888 .888888888. `88888. `8b.  ;8.`8888 8 8888   `Y8.                           
                                    8 8888.8'       `8. `88888. `Y8888P ,88P' 8 8888     `Y8.                         
                                                                                                                      
8 888888888o.   8 8888888888       ,o888888o.        ,o888888o.     8 888888888o.      8 8888888888   8 888888888o.   
8 8888    `88.  8 8888            8888     `88.   . 8888     `88.   8 8888    `^888.   8 8888         8 8888    `88.  
8 8888     `88  8 8888         ,8 8888       `8. ,8 8888       `8b  8 8888        `88. 8 8888         8 8888     `88  
8 8888     ,88  8 8888         88 8888           88 8888        `8b 8 8888         `88 8 8888         8 8888     ,88  
8 8888.   ,88'  8 888888888888 88 8888           88 8888         88 8 8888          88 8 888888888888 8 8888.   ,88'  
8 888888888P'   8 8888         88 8888           88 8888         88 8 8888          88 8 8888         8 888888888P'   
8 8888`8b       8 8888         88 8888           88 8888        ,8P 8 8888         ,88 8 8888         8 8888`8b       
8 8888 `8b.     8 8888         `8 8888       .8' `8 8888       ,8P  8 8888        ,88' 8 8888         8 8888 `8b.     
8 8888   `8b.   8 8888            8888     ,88'   ` 8888     ,88'   8 8888    ,o88P'   8 8888         8 8888   `8b.   
8 8888     `88. 8 888888888888     `8888888P'        `8888888P'     8 888888888P'      8 888888888888 8 8888     `88. 
    ''')
    app = QApplication(sys.argv)  # 创建应用程序对象
    window = MainWindow()  # 创建主窗口
    window.show()  # 显示主窗口
    sys.exit(app.exec())  # 进入消息循环


if __name__ == "__main__":
    main()

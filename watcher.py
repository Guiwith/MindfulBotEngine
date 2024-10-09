import os
import time
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import schedule

# 设置要监控的文件夹路径为当前脚本所在的文件夹
folder_to_monitor = os.path.dirname(os.path.abspath(__file__))

# 获取当前脚本文件名
current_script = os.path.basename(__file__)

# 文件夹监控事件处理类
class FolderMonitorHandler(FileSystemEventHandler):
    def on_modified(self, event):
        print(f"{event.src_path} 被修改")
    def on_created(self, event):
        print(f"{event.src_path} 被创建")
    def on_deleted(self, event):
        print(f"{event.src_path} 被删除")

# 清空文件夹函数
def clear_folder():
    for filename in os.listdir(folder_to_monitor):
        file_path = os.path.join(folder_to_monitor, filename)

        # 跳过当前脚本文件
        if filename == current_script:
            continue

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            print(f"{file_path} 被删除")
        except Exception as e:
            print(f"删除 {file_path} 时出错: {e}")

# 定时任务，每24小时清空文件夹
schedule.every(24).hours.do(clear_folder)

# 监控文件夹
event_handler = FolderMonitorHandler()
observer = Observer()
observer.schedule(event_handler, folder_to_monitor, recursive=False)
observer.start()

try:
    while True:
        # 执行定时任务
        schedule.run_pending()
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()

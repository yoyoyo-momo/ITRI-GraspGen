import tkinter as tk
from DXL import DynamixelController
from actions import ACTIONS
import threading

# ==============================
# 初始化 Dynamixel
# ==============================
dxl = DynamixelController(device_name="/dev/ttyUSB0")

# ==============================
# 動作包裝（避免 UI 卡死）
# ==============================


def run_in_thread(func):
    threading.Thread(target=func, daemon=True).start()


def enable_torque():
    dxl.enable_torque()
    dxl.set_profile(acc=100, vel=150, cur=150)
    # dxl.set_goal_currents({
    #     1: 45,
    #     2: 45,
    #     3: 45,
    #     4: 45,
    #     5: 45,
    #     6: 45,
    #     7: 45,
    #     8: 45
    # })
    print("✅ Torque Enabled")


def disable_torque():
    dxl.disable_torque()
    print("⛔ Torque Disabled")


def open():
    # print("🟢 Open")
    # dxl.set_profile(acc=1000, vel=1000)
    dxl.move_to_positions(ACTIONS["open"])


def hook():
    # print("✊ hook")
    # dxl.set_profile(acc=1000, vel=1000)
    dxl.move_to_positions(ACTIONS["hook"])


def aid():
    # print("✊ aid")
    # dxl.set_profile(acc=1000, vel=1000)
    dxl.move_to_positions(ACTIONS["aid"])


def grasp():
    # print("✊ grasp")
    # dxl.set_profile(acc=1000, vel=1000)
    dxl.move_to_positions(ACTIONS["grasp"])


def fist():
    # print("✊ grasp")
    # dxl.set_profile(acc=1000, vel=1000)
    dxl.move_to_positions(ACTIONS["fist"])


# ==============================
# UI
# ==============================
root = tk.Tk()
root.title("Dynamixel Hand Control")
root.geometry("300x300")

btn_font = ("Arial", 14)

tk.Button(
    root,
    text="ENABLE",
    font=btn_font,
    bg="lightgreen",
    command=lambda: run_in_thread(enable_torque),
).pack(fill="x", pady=5)

tk.Button(
    root,
    text="DISABLE",
    font=btn_font,
    bg="lightcoral",
    command=lambda: run_in_thread(disable_torque),
).pack(fill="x", pady=5)

tk.Button(root, text="OPEN", font=btn_font, command=lambda: run_in_thread(open)).pack(
    fill="x", pady=5
)

tk.Button(root, text="HOOK", font=btn_font, command=lambda: run_in_thread(hook)).pack(
    fill="x", pady=5
)

tk.Button(root, text="AID", font=btn_font, command=lambda: run_in_thread(aid)).pack(
    fill="x", pady=5
)

tk.Button(root, text="GRASP", font=btn_font, command=lambda: run_in_thread(grasp)).pack(
    fill="x", pady=5
)

tk.Button(root, text="FIST", font=btn_font, command=lambda: run_in_thread(fist)).pack(
    fill="x", pady=5
)


# ==============================
# 關閉處理
# ==============================
def on_close():
    print("🔚 Closing...")
    dxl.close()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()

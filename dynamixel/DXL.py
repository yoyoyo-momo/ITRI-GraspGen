from dynamixel_sdk import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)
import time
import threading


class DynamixelController:
    def __init__(
        self,
        device_name="COM3",
        baudrate=57600,
        dxl_ids=None,
        protocol_version=2.0,
    ):
        # ==============================
        # 基本參數
        # ==============================
        self.DEVICENAME = device_name
        self.BAUDRATE = baudrate
        if dxl_ids is None:
            dxl_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        self.DXL_IDS = dxl_ids
        self.PROTOCOL_VERSION = protocol_version

        # Control Table
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_PROFILE_ACCELERATION = 108
        self.ADDR_PROFILE_VELOCITY = 112
        self.ADDR_GOAL_CURRENT = 102
        self.ADDR_GOAL_POSITION = 116

        self.ADDR_PRESENT_POSITION = 132
        self.ADDR_PRESENT_CURRENT = 126

        self.LEN_PRESENT_POSITION = 4
        self.LEN_PRESENT_CURRENT = 2

        self.TORQUE_ENABLE = 1
        self.TORQUE_DISABLE = 0

        self.MOVING_THRESHOLD = 50

        self.CURRENT_DEADBAND = 20  # raw unit，避免抖動
        self.goal_currents = {}  # {dxl_id: goal_current}

        self.ADDR_HARDWARE_ERROR = 70
        self.ADDR_PRESENT_TEMPERATURE = 146
        self.ADDR_PRESENT_VOLTAGE = 144

        self.ADDR_POS_D_GAIN = 80
        self.ADDR_POS_I_GAIN = 82
        self.ADDR_POS_P_GAIN = 84
        self.LEN_PID_ALL = 6

        # ==============================
        # 初始化 Port / Packet
        # ==============================
        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        if not self.portHandler.openPort():
            raise RuntimeError("❌ 無法開啟 COM Port")

        if not self.portHandler.setBaudRate(self.BAUDRATE):
            raise RuntimeError("❌ 無法設定 Baudrate")

        # ==============================
        # Sync Read / Write
        # ==============================
        # 位置同步讀取
        self.syncReadPos = GroupSyncRead(
            self.portHandler,
            self.packetHandler,
            self.ADDR_PRESENT_POSITION,
            self.LEN_PRESENT_POSITION,
        )
        for dxl_id in self.DXL_IDS:
            self.syncReadPos.addParam(dxl_id)

        # ✅ 電流同步讀取
        self.syncReadCur = GroupSyncRead(
            self.portHandler,
            self.packetHandler,
            self.ADDR_PRESENT_CURRENT,
            self.LEN_PRESENT_CURRENT,
        )
        for dxl_id in self.DXL_IDS:
            self.syncReadCur.addParam(dxl_id)

        # torque sync write
        self.syncWriteTorque = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.ADDR_TORQUE_ENABLE, 1
        )

        # profile acc sync write
        self.syncWriteAcc = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.ADDR_PROFILE_ACCELERATION, 4
        )

        # profile vel sync write
        self.syncWriteVel = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.ADDR_PROFILE_VELOCITY, 4
        )

        # profile cur sync write
        self.syncWriteCur = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.ADDR_GOAL_CURRENT, 2
        )

        # goal pos sync write
        self.syncWritePos = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.ADDR_GOAL_POSITION, 4
        )

        # pid sync write
        # 建立一個從位址 80 (D Gain) 開始，長度為 6 Byte 的同步寫入物件
        # (2 byte D + 2 byte I + 2 byte P)
        self.syncWritePID = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.ADDR_POS_D_GAIN, self.LEN_PID_ALL
        )

        # pid sync read
        # 建立一個從位址 80 (D Gain) 開始，長度為 6 Byte 的同步讀取物件
        # (2 byte D + 2 byte I + 2 byte P)
        self.syncReadPID = GroupSyncRead(
            self.portHandler, self.packetHandler, self.ADDR_POS_D_GAIN, self.LEN_PID_ALL
        )
        for dxl_id in self.DXL_IDS:
            self.syncReadPID.addParam(dxl_id)

    # ==============================
    # 小工具：uint32 -> int32
    # 小工具：uint16 -> int16
    # ==============================
    @staticmethod
    def _int32_from_uint32(x: int) -> int:
        """Present Velocity 是 signed int32，轉換後方向才正確"""
        if x >= 0x80000000:
            return x - 0x100000000
        return x

    @staticmethod
    def _int16_from_uint16(x: int) -> int:
        if x >= 0x8000:
            return x - 0x10000
        return x

    # ==============================
    # Torque Control
    # ==============================
    def enable_torque(self):
        for dxl_id in self.DXL_IDS:
            self.syncWriteTorque.addParam(dxl_id, [self.TORQUE_ENABLE])
        self.syncWriteTorque.txPacket()
        self.syncWriteTorque.clearParam()

    def disable_torque(self):
        for dxl_id in self.DXL_IDS:
            self.syncWriteTorque.addParam(dxl_id, [self.TORQUE_DISABLE])
        self.syncWriteTorque.txPacket()
        self.syncWriteTorque.clearParam()

    def reset_motors(self):
        for dxl_id in self.DXL_IDS:
            self.packetHandler.reboot(self.portHandler, dxl_id)
            time.sleep(0.1)
        # time.sleep(2)

    # ==============================
    # Profile 設定
    # ==============================
    def set_PID(self, p, i, d):
        """
        利用 SyncWrite 同步設定所有馬達的 PID
        """
        for dxl_id in self.DXL_IDS:
            param = [
                DXL_LOBYTE(d),
                DXL_HIBYTE(d),  # Address 80-81: D
                DXL_LOBYTE(i),
                DXL_HIBYTE(i),  # Address 82-83: I
                DXL_LOBYTE(p),
                DXL_HIBYTE(p),  # Address 84-85: P
            ]
            self.syncWritePID.addParam(dxl_id, param)

        self.syncWritePID.txPacket()
        self.syncWritePID.clearParam()
        """
        讀取並回傳目前馬達的 PID 數值
        """
        self.syncReadPID.txRxPacket()
        for dxl_id in self.DXL_IDS:
            d_val = self.syncReadPID.getData(dxl_id, 80, 2)
            i_val = self.syncReadPID.getData(dxl_id, 82, 2)
            p_val = self.syncReadPID.getData(dxl_id, 84, 2)
            # print(f"ID:{dxl_id} -> P:{p_val}, I:{i_val}, D:{d_val}")
        print(f"P:{p_val}, I:{i_val}, D:{d_val}")

    def set_profile(self, acc=50, vel=200, cur=300):
        # Acc
        for dxl_id in self.DXL_IDS:
            acc_param = [
                DXL_LOBYTE(DXL_LOWORD(acc)),
                DXL_HIBYTE(DXL_LOWORD(acc)),
                DXL_LOBYTE(DXL_HIWORD(acc)),
                DXL_HIBYTE(DXL_HIWORD(acc)),
            ]
            self.syncWriteAcc.addParam(dxl_id, acc_param)

        self.syncWriteAcc.txPacket()
        self.syncWriteAcc.clearParam()

        # Vel
        for dxl_id in self.DXL_IDS:
            vel_param = [
                DXL_LOBYTE(DXL_LOWORD(vel)),
                DXL_HIBYTE(DXL_LOWORD(vel)),
                DXL_LOBYTE(DXL_HIWORD(vel)),
                DXL_HIBYTE(DXL_HIWORD(vel)),
            ]
            self.syncWriteVel.addParam(dxl_id, vel_param)

        self.syncWriteVel.txPacket()
        self.syncWriteVel.clearParam()

        # cul
        for dxl_id in self.DXL_IDS:
            cur_param = [DXL_LOBYTE(cur), DXL_HIBYTE(cur)]
            self.syncWriteCur.addParam(dxl_id, cur_param)

        self.syncWriteCur.txPacket()
        self.syncWriteCur.clearParam()

        print("加速度:", acc, "速度:", vel, "電流:", cur)

    # ==============================
    # 取得目前位置（SyncRead）
    # ==============================
    def get_positions(self):
        positions = {}
        self.syncReadPos.txRxPacket()

        for dxl_id in self.DXL_IDS:
            if self.syncReadPos.isAvailable(
                dxl_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION
            ):
                pos = self.syncReadPos.getData(
                    dxl_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION
                )
                positions[dxl_id] = pos
            else:
                positions[dxl_id] = None

        return positions.values()

    # ==============================
    # ✅ 取得目前電流（SyncRead）
    # ==============================
    def get_currents(self):
        currents = {}

        dxl_comm_result = self.syncReadCur.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print(
                "❌ SyncReadCur 失敗:",
                self.packetHandler.getTxRxResult(dxl_comm_result),
            )
            for dxl_id in self.DXL_IDS:
                currents[dxl_id] = "N/A"
            return currents

        for dxl_id in self.DXL_IDS:
            if self.syncReadCur.isAvailable(
                dxl_id, self.ADDR_PRESENT_CURRENT, self.LEN_PRESENT_CURRENT
            ):
                raw = self.syncReadCur.getData(
                    dxl_id, self.ADDR_PRESENT_CURRENT, self.LEN_PRESENT_CURRENT
                )
                cur = self._int16_from_uint16(raw)
                currents[dxl_id] = cur
            else:
                currents[dxl_id] = "N/A"

        return currents

    # ==============================
    # 移動到指定位置 (array)
    # ==============================

    def move_to_positions(
        self,
        positions_array,
        # check_interval=0.01
    ):
        if len(positions_array) != len(self.DXL_IDS):
            raise ValueError("positions_array 長度必須等於馬達數量")

        goal_positions = dict(zip(self.DXL_IDS, positions_array, strict=True))

        # ===== SyncWrite Goal Position =====
        for dxl_id, pos in goal_positions.items():
            param = [
                DXL_LOBYTE(DXL_LOWORD(pos)),
                DXL_HIBYTE(DXL_LOWORD(pos)),
                DXL_LOBYTE(DXL_HIWORD(pos)),
                DXL_HIBYTE(DXL_HIWORD(pos)),
            ]
            self.syncWritePos.addParam(dxl_id, param)

        self.syncWritePos.txPacket()
        self.syncWritePos.clearParam()

    def safe_move(self, positions_array):
        self.auto_recover()
        self.move_to_positions(positions_array)

    # ==============================
    # 關閉
    # ==============================
    def close(self):
        self.disable_torque()
        self.portHandler.closePort()

    # ==============================
    # error debug
    # ==============================
    def decode_error(self, err):
        errors = []

        if err & 1:
            errors.append("Input Voltage")
        if err & 4:
            errors.append("Overheating")
        if err & 8:
            errors.append("Encoder")
        if err & 16:
            errors.append("Electrical Shock")
        if err & 32:
            errors.append("Overload")

        return errors

    def check_hardware_errors(self):
        error_motors = []

        for dxl_id in self.DXL_IDS:
            err, comm, _ = self.packetHandler.read1ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_HARDWARE_ERROR
            )

            if comm != COMM_SUCCESS:
                print("Comm Error:", self.packetHandler.getTxRxResult(comm))
                continue

            if err != 0:
                decoded = self.decode_error(err)

                print(f"⚠ Motor {dxl_id} error:", decoded)

                error_motors.append(dxl_id)

        return error_motors

    def auto_recover(self):
        error_motors = self.check_hardware_errors()

        if len(error_motors) == 0:
            return

        print("🔧 Recovering motors:", error_motors)

        for dxl_id in error_motors:
            self.packetHandler.reboot(self.portHandler, dxl_id)
            time.sleep(0.3)

        time.sleep(1)

        # 重新 enable torque
        self.enable_torque()

    def start_watchdog(self):
        def monitor():
            while True:
                errors = self.check_hardware_errors()

                if errors:
                    self.auto_recover()

                time.sleep(0.5)

        t = threading.Thread(target=monitor, daemon=True)
        t.start()


if __name__ == "__main__":
    dxl = DynamixelController(device_name="/dev/dynamixel")

    # dxl.reset_motors()
    # dxl.enable_torque()
    dxl.set_PID(2100, 0, 1900)
    # dxl.set_profile(acc=50, vel=80, cur=100)

    # dxl.set_goal_currents({
    #     1: 200,
    #     2: 220,
    #     3: 180,
    #     4: 200,
    #     5: 250,
    #     6: 250,
    #     7: 200,
    #     8: 200
    # })

    print("目前位置:", dxl.get_positions())
    # time.sleep(2)
    # dxl.move_to_positions(ACTIONS["open"], wait=True, current_limit=200)
    # time.sleep(2)
    # dxl.move_to_positions(ACTIONS["hook"], wait=True, current_limit=200)
    # time.sleep(2)
    # dxl.move_to_positions(ACTIONS["aid"], wait=True, current_limit=200)
    # time.sleep(5)
    # 動作中連續印速度
    # dxl.move_to_positions(MOTIONS["close"])
    # for _ in range(30):
    #     print("目前速度:", dxl.get_velocities())
    #     time.sleep(0.01)

    # dxl.close()

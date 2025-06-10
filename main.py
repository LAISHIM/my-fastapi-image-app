from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Request,
    Depends,
    Form,
    status,
)
from fastapi.responses import (
    StreamingResponse,
    HTMLResponse,
    FileResponse,
    RedirectResponse,
    JSONResponse,  # 导入 JSONResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware  # 导入 CORS 中间件

import cv2 as cv  # 将cv2导入为cv
import asyncio
import threading
from typing import Generator, Optional
import time
import numpy as np
import os
import timeit
import uuid
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from urllib.parse import urlparse, urlunparse, parse_qs, quote_plus, unquote_plus

# JWT 相关的导入
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone

# 加载环境变量
load_dotenv()

# --- FastAPI 应用初始化 ---
app = FastAPI(
    title="智能图像处理与认证系统",
    description="本地摄像头视频流、图片处理和用户认证服务",
)

# --- CORS 配置 ---
# origins 列表应包含您的前端应用所部署的所有域名
# 确保包含您的 GitHub Pages 域名，且大小写正确
origins = [
    "http://localhost:8000",  # 本地开发测试用
    "http://127.0.0.1:8000",  # 本地开发测试用
    "https://LAISHIM.github.io",  # <-- 已修正为大写 LAISHIM
    # 如果您的前端部署在其他路径，例如 https://LAISHIM.github.io/your-repo-name/
    # 您可能需要根据实际情况调整此列表。
    # 如果 Replit 后端有自己的访问域名，也可能需要加进来
    # "https://3571b520-68af-404d-828d-860db9009f16-00-3w177559c5lwh.sisko.replit.dev", # 如果你的replit部署域名也需要CORS
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # 允许发送 cookie (对于 JWT 认证很重要)
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有 HTTP 头
)

# --- JWT 配置 ---
SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "your-super-secret-jwt-key-that-you-must-change")  # 从环境变量获取密钥
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Access Token 有效期（分钟）

# --- 邮件服务配置 ---
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))  # 确保端口是整数
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# --- 数据库配置 ---
DATABASE_URL = "email.db"  # SQLite 数据库文件

# --- 视频流相关全局变量 ---
# 这是一个模拟的视频源帧，可以替换为实际的摄像头捕获
# 例如：camera = cv2.VideoCapture(0) # 0 表示默认摄像头
camera = None  # 摄像头对象，初始为None
frame_lock = threading.Lock()  # 帧锁，用于多线程同步
current_frame = None  # 当前帧，用于存储摄像头捕获的最新帧
last_capture_time = 0  # 上次捕获帧的时间
CAPTURE_INTERVAL = 0.1  # 捕获帧的间隔（秒），例如10fps

# 人脸和眼睛检测模型（OpenCV Haar Cascades）
# 确保 these XML files are accessible relative to your main.py or from cv2.data
face_cascade = cv.CascadeClassifier(cv.data.haarcascades +
                                    "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades +
                                   "haarcascade_eye.xml"  # <--- 这里就是修改的地方
                                   )

# --- 目录配置 ---
# 确保这些目录存在
UPLOAD_DIR = "static/uploads"
PROCESSED_DIR = "static/processed"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 配置 Jinja2 模板
templates = Jinja2Templates(directory="templates")


# --- 数据库操作 ---
def init_db():
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            verification_code TEXT,
            code_expires_at TEXT,
            is_verified INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    print("应用启动中...")
    init_db()  # 初始化数据库
    # 在应用启动时尝试打开摄像头
    # global camera # 如果使用真实摄像头，取消注释
    # camera = cv.VideoCapture(0) # 如果使用真实摄像头，取消注释
    # if camera.isOpened(): # 如果使用真实摄像头，取消注释
    #       print("摄像头已成功打开。") # 如果使用真实摄像头，取消注释
    # else: # 如果使用真实摄像头，取消注释
    #       print("警告：无法打开摄像头。视频流将不可用。") # 如果使用真实摄像头，取消注释
    # asyncio.create_task(update_frame()) # 如果使用真实摄像头，取消注释
    await run_performance_test()  # 运行性能测试
    yield
    # 关闭时执行
    print("应用关闭中...")
    # if camera and camera.isOpened(): # 如果使用真实摄像头，取消注释
    #       camera.release() # 如果使用真实摄像头，取消注释
    #       print("摄像头已关闭。") # 如果使用真实摄像头，取消注释


app.router.lifespan_context = lifespan


# --- JWT 辅助函数 ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


async def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER,
                            headers={"Location": "/login"})
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER,
                            headers={"Location": "/login"})
    user_email = payload.get("sub")
    if user_email is None:
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER,
                            headers={"Location": "/login"})

    # 检查用户是否已验证
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT is_verified FROM users WHERE email = ?",
                   (user_email, ))
    result = cursor.fetchone()
    conn.close()

    if not result or result[0] != 1:  # 如果用户不存在或未验证
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER,
                            headers={"Location": "/login"})
    return user_email


# --- 邮件发送函数 ---
async def send_verification_email(email: str, code: str):
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD]):
        print("SMTP 配置不完整，无法发送邮件。请检查环境变量。")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="邮件服务配置错误")

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = email
    msg["Subject"] = "您的验证码"

    body = f"您的验证码是: {code}\n请在 5 分钟内使用此验证码进行登录。"
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"验证码已发送到: {email}")
    except Exception as e:
        print(f"发送邮件失败: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="发送验证码失败")


# --- 路由定义 ---


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request,
                    user_email: str = Depends(get_current_user)):
    # print(f"User {user_email} accessed index page.")
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_email": user_email
    })


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    # 如果用户已经登录，重定向到主页
    if request.cookies.get("access_token"):
        payload = decode_access_token(request.cookies.get("access_token"))
        if payload and payload.get("sub"):
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()
            cursor.execute("SELECT is_verified FROM users WHERE email = ?",
                           (payload.get("sub"), ))
            result = cursor.fetchone()
            conn.close()
            if result and result[0] == 1:
                return RedirectResponse(url="/",
                                        status_code=status.HTTP_302_FOUND)

    # 获取重定向目标（如果有）
    query_params = parse_qs(urlparse(str(request.url)).query)
    redirect_to = query_params.get("redirect_to", ["/"])[0]

    return templates.TemplateResponse("login.html", {
        "request": request,
        "redirect_to": redirect_to
    })


@app.post("/send-code")
async def send_code(request: Request, email: str = Form(...)):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    code = str(random.randint(100000, 999999))
    expires_at = (datetime.now(timezone.utc) +
                  timedelta(minutes=5)).isoformat()

    cursor.execute("SELECT email FROM users WHERE email = ?", (email, ))
    if cursor.fetchone():
        cursor.execute(
            "UPDATE users SET verification_code = ?, code_expires_at = ?, is_verified = 0 WHERE email = ?",
            (code, expires_at, email),
        )
    else:
        cursor.execute(
            "INSERT INTO users (email, verification_code, code_expires_at) VALUES (?, ?, ?)",
            (email, code, expires_at),
        )
    conn.commit()
    conn.close()

    try:
        await send_verification_email(email, code)
        return JSONResponse(content={"message": "验证码已发送到您的邮箱。"},
                            status_code=status.HTTP_200_OK)
    except HTTPException as e:
        # 捕获 send_verification_email 抛出的 HTTPException
        return JSONResponse(content={"detail": e.detail},
                            status_code=e.status_code)
    except Exception as e:
        # 其他未知错误
        return JSONResponse(content={"detail": "发送验证码失败: " + str(e)},
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post("/login")
async def process_login(request: Request,
                        email: str = Form(...),
                        code: str = Form(...),
                        redirect_to: str = Form(...)):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT verification_code, code_expires_at FROM users WHERE email = ?",
        (email, ))
    user_data = cursor.fetchone()
    conn.close()

    if not user_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="邮箱未注册或验证码错误。")

    stored_code, expires_at_str = user_data
    expires_at = datetime.fromisoformat(expires_at_str)

    if stored_code != code or datetime.now(timezone.utc) > expires_at:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="验证码错误或已过期。")

    # 验证成功，更新用户状态
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET is_verified = 1 WHERE email = ?",
                   (email, ))
    conn.commit()
    conn.close()

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": email},
                                       expires_delta=access_token_expires)

    # 创建响应并设置 Cookie
    response = RedirectResponse(url=redirect_to,
                                status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # max_age in seconds
        expires=datetime.now(timezone.utc) +
        access_token_expires,  # expires in GMT for older browsers
        path="/",
        secure=True,  # 生产环境请打开此项，因为您的前端是 HTTPS
        samesite="Lax"  # 推荐 Lax 或 Strict，防止 CSRF
    )
    return response


@app.get("/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/login",
                                status_code=status.HTTP_302_FOUND)
    response.delete_cookie("access_token", path="/")
    return response


@app.post("/upload_image/")
async def upload_image(request: Request,
                       file: UploadFile = File(...),
                       user_email: str = Depends(get_current_user)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只允许上传图片文件。")

    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    original_filepath = os.path.join(UPLOAD_DIR, unique_filename)
    processed_filepath = os.path.join(PROCESSED_DIR,
                                      f"processed_{unique_filename}")

    # 保存原始图片
    with open(original_filepath, "wb") as buffer:
        buffer.write(await file.read())

    # 读取图片并进行灰度处理
    img = cv.imread(original_filepath)
    if img is None:
        raise HTTPException(status_code=500, detail="无法读取上传的图片。")

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 保存处理后的图片
    cv.imwrite(processed_filepath, gray_img)

    # 构建完整的 URL
    # 使用 request.url_for 和 request.base_url 动态生成完整 URL
    # 原始图片 URL
    original_image_url = request.url_for("static",
                                         path=f"uploads/{unique_filename}")
    # 处理后的图片 URL
    processed_image_url = request.url_for(
        "static", path=f"processed/processed_{unique_filename}")

    # 获取完整的 base URL，并确保它是正确的协议 (https)
    base_url = str(request.base_url)
    if not base_url.startswith("https://"):
        parsed_url = urlparse(base_url)
        # 如果是 http，尝试改为 https
        if parsed_url.scheme == "http":
            base_url = urlunparse(parsed_url._replace(scheme="https"))
        # 否则，如果不是 http 也不是 https (比如 file://)，则可能需要特殊处理或报错
        else:
            # 对于这种情况，如果直接用 request.base_url 拼接可能不安全
            # 更安全的做法是依赖 Fastapi 的 url_for 自动生成绝对路径
            pass

    # 如果 request.url_for 返回的是相对路径，则需要拼接 base_url
    if not original_image_url.startswith("http"):
        original_image_url = f"{base_url}{original_image_url}"
    if not processed_image_url.startswith("http"):
        processed_image_url = f"{base_url}{processed_image_url}"

    return JSONResponse(
        content={
            "message": "图片处理成功！",
            "original_image_url": original_image_url,
            "processed_image_url": processed_image_url,
        })


# 模拟视频流生成器
def generate_frames():
    # 如果没有真实的摄像头，可以使用一个示例图片循环生成
    # 或者留空，让客户端知道没有视频流
    # 这里我们使用一个简单的黑色帧作为占位符
    if camera and camera.isOpened():
        while True:
            with frame_lock:
                if current_frame is not None:
                    ret, buffer = cv.imencode(".jpg", current_frame)
                    if ret:
                        yield (b"--frame\r\n"
                               b"Content-Type: image/jpeg\r\n\r\n" +
                               buffer.tobytes() + b"\r\n")
            time.sleep(0.03)  # 控制帧率
    else:
        # 如果摄像头未打开，生成一个表示无视频的静态图片
        # 比如一个带有文字的黑色图片
        width, height = 640, 480
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        text = "No Camera Feed"
        font = cv.FONT_HERSHEY_SIMPLEX
        text_size = cv.getTextSize(text, font, 1.5, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv.putText(dummy_frame, text, (text_x, text_y), font, 1.5,
                   (255, 255, 255), 2, cv.LINE_AA)

        ret, buffer = cv.imencode(".jpg", dummy_frame)
        if ret:
            while True:  # 持续发送这个静态帧
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() +
                       b"\r\n")
                time.sleep(1)  # 不需要太快，因为它不会变化


# 如果使用真实摄像头，以下函数将用于持续更新 `current_frame`
async def update_frame():
    global current_frame, last_capture_time
    while True:
        if camera and camera.isOpened():
            current_time = time.time()
            if current_time - last_capture_time > CAPTURE_INTERVAL:
                ret, frame = camera.read()
                if not ret:
                    print("无法从摄像头读取帧。")
                    break
                # 在这里可以对帧进行处理，例如人脸检测
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh),
                                     (0, 255, 0), 2)

                with frame_lock:
                    current_frame = frame
                last_capture_time = current_time
        await asyncio.sleep(0.01)  # 短暂休眠以避免CPU过高占用


@app.get("/video_feed")
async def video_feed(request: Request,
                     user_email: str = Depends(get_current_user)):
    # print(f"User {user_email} requested video feed.")
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")


# 性能测试函数
async def run_performance_test():
    test_image_path = "static/test_image.jpg"  # 确保您有这个测试图片

    # 检查测试图片是否存在
    if os.path.exists(test_image_path):
        try:
            img_for_test = cv.imread(test_image_path)
            if img_for_test is not None:
                print("\n--- 图像处理性能测试 ---")

                # 测试 NumPy 方法
                def numpy_gray_value_test():
                    r = img_for_test[:, :, 2]
                    g = img_for_test[:, :, 1]
                    b = img_for_test[:, :, 0]
                    gray_value = 0.299 * r + 0.587 * g + 0.114 * b
                    return gray_value

                # 测试 OpenCV 方法
                def cv2_gray_value_test():
                    gray_value = cv.cvtColor(img_for_test, cv.COLOR_BGR2GRAY)
                    return gray_value

                result2 = timeit.timeit(numpy_gray_value_test, number=100)
                result3 = timeit.timeit(cv2_gray_value_test, number=100)

                print(
                    f"time for numpy_gray_value_test (vectorized): {result2:.6f} seconds"
                )
                print(f"time for cv2_gray_value_test: {result3:.6f} seconds")
                print("--- 性能测试结束 ---\n")
            else:
                print(f"警告: 无法读取用于性能测试的图片: '{test_image_path}'. 请确保文件存在且可读。")
        except Exception as e:
            print(f"性能测试过程中发生错误: {e}")
    else:
        print(f"警告: 用于性能测试的图片 '{test_image_path}' 不存在。跳过性能测试。")

    print("启动服务器...")
    print("登录页面地址: http://localhost:8000/login")
    print("视频流地址: http://localhost:8000/video_feed")

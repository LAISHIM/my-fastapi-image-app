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
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware  # 导入 CORS 中间件

import cv2
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
from urllib.parse import urlparse, parse_qs, quote_plus, unquote_plus

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
# 确保包含您的 GitHub Pages 域名
origins = [
    "http://localhost:8000",  # 本地开发测试用
    "http://127.0.0.1:8000",  # 本地开发测试用
    "https://laishim.github.io",  # 您的 GitHub Pages 域名
    # 如果您有其他前端域名，请在此处添加，例如：
    # "https://www.your-custom-domain.com",
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
    "JWT_SECRET_KEY", "your-super-secret-jwt-key-that-you-must-change"
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # JWT 访问令牌的有效期（分钟）

if SECRET_KEY == "your-super-secret-jwt-key-that-you-must-change":
    print("警告: JWT_SECRET_KEY 使用默认值，请在 .env 文件中设置一个强随机字符串。")

# --- 邮件配置 (从 .env 加载) ---
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# --- 数据库配置 ---
DATABASE_FILE = "email.db"
VERIFICATION_CODE_EXPIRY_SECONDS = 10 * 60  # 10 分钟
EMAIL_SEND_INTERVAL_SECONDS = 60  # 1 分钟

# --- 图片处理相关配置和目录创建 ---
UPLOAD_FOLDER = "uploaded_images"
PROCESSED_FOLDER = "processed_images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --- 数据库操作函数 (保持不变) ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def create_email_code_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS email_code (
            "to" TEXT NOT NULL,
            "code" TEXT NOT NULL,
            "expire_time" INTEGER NOT NULL,
            "send_time" INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()
    print("数据库表 email_code 检查并创建成功")


def upsert_email_code(to: str, code: str, expire_time: int, send_time: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        now = int(time.time())
        cursor.execute(
            'DELETE FROM email_code WHERE "to" = ? AND "expire_time" > ?', (to, now)
        )
        cursor.execute(
            'INSERT INTO email_code ("to", "code", "expire_time", "send_time") VALUES (?, ?, ?, ?)',
            (to, code, expire_time, send_time),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"插入/更新验证码失败: {e}")
        return False


def check_email_code_valid(to: str, code: str):
    now = int(time.time())
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM email_code WHERE "to" = ? AND "code" = ? AND "expire_time" > ? ORDER BY expire_time DESC LIMIT 1',
        (to, code, now),
    )
    result = cursor.fetchone()
    conn.close()
    return result is not None


def check_email_send_limit(to: str, interval_seconds: int):
    now = int(time.time())
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM email_code WHERE "to" = ? AND "send_time" > ?',
        (to, now - interval_seconds),
    )
    result = cursor.fetchone()
    conn.close()
    return result is not None


def send_email(to: str, code: str):
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD]):
        print("邮件配置不完整，无法发送邮件。请检查 .env 文件。")
        return False

    subject = "来自智能图像处理系统的验证码"
    body = f"""
    <h1>{subject}</h1>
    <p>您的验证码是：<strong>{code}</strong></p>
    <p>此验证码将在 {VERIFICATION_CODE_EXPIRY_SECONDS // 60} 分钟内过期。请尽快使用。</p>
    <p>如果您没有请求此验证码，请忽略此邮件。</p>
    """

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "html", "utf-8"))

    try:
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, to, msg.as_string())
        server.quit()
        print(f"邮件发送成功到: {to}")
        return True
    except smtplib.SMTPAuthenticationError:
        print(f"邮件发送失败到 {to}: SMTP 认证失败，请检查邮箱用户名和密码。")
        return False
    except smtplib.SMTPServerDisconnected:
        print(f"邮件发送失败到 {to}: SMTP 服务器连接断开。")
    except Exception as e:
        print(f"邮件发送失败到 {to}: {e}")
        return False


# --- JWT 认证函数 ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# JWT 依赖注入
async def get_current_user_jwt(request: Request) -> str:
    # 尝试从 cookie 中获取 token
    token = request.cookies.get("access_token")
    if not token:
        # 如果 cookie 中没有，尝试从 Authorization header 中获取 (例如用于 API 调用)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

    if not token:
        # 如果没有 token，则未认证，重定向到登录页
        full_path_for_redirect = request.url.path
        if request.url.query:
            full_path_for_redirect += f"?{request.url.query}"

        parsed_url_for_redirect = urlparse(full_path_for_redirect)
        query_params = parse_qs(parsed_url_for_redirect.query)
        if "redirect_to" in query_params:
            del query_params["redirect_to"]

        new_query_string_for_redirect = "&".join(
            [f"{k}={v[0]}" for k, v in query_params.items()]
        )
        reconstructed_path_for_redirect = parsed_url_for_redirect._replace(
            query=new_query_string_for_redirect
        ).geturl()

        redirect_url_with_param = (
            request.url_for("login_page")
            + f"?redirect_to={quote_plus(reconstructed_path_for_redirect)}"
        )

        raise HTTPException(
            status_code=status.HTTP_302_FOUND,
            detail="Not authenticated",
            headers={"Location": redirect_url_with_param},
        )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email: str = payload.get("sub")  # 'sub' 是 JWT 规范中用于主题的字段
        if user_email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return user_email
    except JWTError:
        # JWT 验证失败 (无效签名, 过期等)
        full_path_for_redirect = request.url.path
        if request.url.query:
            full_path_for_redirect += f"?{request.url.query}"

        parsed_url_for_redirect = urlparse(full_path_for_redirect)
        query_params = parse_qs(parsed_url_for_redirect.query)
        if "redirect_to" in query_params:
            del query_params["redirect_to"]

        new_query_string_for_redirect = "&".join(
            [f"{k}={v[0]}" for k, v in query_params.items()]
        )
        reconstructed_path_for_redirect = parsed_url_for_redirect._replace(
            query=new_query_string_for_redirect
        ).geturl()

        redirect_url_with_param = (
            request.url_for("login_page")
            + f"?redirect_to={quote_plus(reconstructed_path_for_redirect)}"
        )

        raise HTTPException(
            status_code=status.HTTP_302_FOUND,
            detail="Token expired or invalid",
            headers={"Location": redirect_url_with_param},
        )


# --- 通用图片处理函数 (保持不变) ---
def allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image_to_grayscale(image_path: str) -> Optional[np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return None
    gray_value = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_value


# --- 视频流和人脸检测代码 (保持不变) ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haascades + "haarcascade_eye.xml")

if face_cascade.empty() or eye_cascade.empty():
    print("警告: 无法加载 Haar 级联分类器。人脸/眼睛检测功能可能无法正常工作。")
    print(f"请检查路径: {cv2.data.haascades}haarcascade_frontalface_default.xml")
    print(f"请检查路径: {cv2.data.haascades}haarcascade_eye.xml")


class VideoStreamer:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.is_streaming = False
        self.lock = threading.Lock()
        self.camera_thread: Optional[threading.Thread] = None

    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"无法打开摄像头 {self.camera_index}。请确保摄像头已连接且未被占用。"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.is_streaming = True
        self.camera_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.camera_thread.start()

    def _read_frames(self):
        while self.is_streaming and self.cap:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if not face_cascade.empty():
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    for x, y, w, h in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        roi_gray = gray[y : y + h, x : x + w]
                        roi_color = frame[y : y + h, x : x + w]
                        if not eye_cascade.empty():
                            eyes = eye_cascade.detectMultiScale(roi_gray)
                            for ex, ey, ew, eh in eyes:
                                cv2.rectangle(
                                    roi_color,
                                    (ex, ey),
                                    (ex + ew, ey + eh),
                                    (0, 255, 0),
                                    2,
                                )
                with self.lock:
                    self.frame = frame
            time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop_camera(self):
        self.is_streaming = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None


video_streamer = VideoStreamer()


def generate_frames() -> Generator[bytes, None, None]:
    while True:
        frame = video_streamer.get_frame()
        if frame is not None:
            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            else:
                print("警告: 无法编码帧为 JPEG。")
        time.sleep(0.033)


# --- 应用生命周期事件钩子 (保持不变) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_email_code_table()
    try:
        video_streamer.start_camera()
        print("摄像头启动成功")
    except Exception as e:
        print(f"摄像头启动失败: {e}")
    yield
    video_streamer.stop_camera()
    print("摄像头已释放")


app.router.lifespan_context = lifespan

# --- 路由定义 ---


@app.get("/login", response_class=HTMLResponse, name="login_page")
async def login_page(request: Request, redirect_to: Optional[str] = None):
    safe_redirect_to = None
    if redirect_to:
        decoded_redirect_to = unquote_plus(redirect_to)
        parsed_url = urlparse(decoded_redirect_to)
        if (
            not parsed_url.netloc
            and not parsed_url.scheme
            and parsed_url.path.startswith("/")
        ):
            safe_redirect_to = decoded_redirect_to
        else:
            print(f"检测到不安全的 redirect_to URL: {redirect_to}，已忽略。")
    return templates.TemplateResponse(
        "login.html", {"request": request, "redirect_to": safe_redirect_to}
    )


@app.post("/send-code")
async def send_verification_code(email: str = Form(...)):
    if not email or "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="邮箱地址格式不正确或为空。")

    if check_email_send_limit(email, EMAIL_SEND_INTERVAL_SECONDS):
        raise HTTPException(
            status_code=429,
            detail=f"发送验证码过于频繁，请等待 {EMAIL_SEND_INTERVAL_SECONDS} 秒后重试。",
        )

    code = "".join(random.choices("0123456789", k=6))
    current_time = int(time.time())
    expire_time = current_time + VERIFICATION_CODE_EXPIRY_SECONDS

    if upsert_email_code(email, code, expire_time, current_time):
        if send_email(email, code):
            return {
                "message": "验证码已发送，请检查您的邮箱。",
                "expiry_minutes": VERIFICATION_CODE_EXPIRY_SECONDS // 60,
            }
        else:
            raise HTTPException(
                status_code=500, detail="邮件发送失败，请检查服务器配置或稍后再试。"
            )
    else:
        raise HTTPException(status_code=500, detail="保存验证码失败，请稍后再试。")


@app.post("/login")
async def process_login(
    request: Request,
    email: str = Form(...),
    code: str = Form(...),
    redirect_to: Optional[str] = None,
):
    if not email or not code:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error_message": "邮箱和验证码不能为空。",
                "redirect_to": redirect_to,
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if check_email_code_valid(email, code):
        # 验证成功，创建 JWT
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": email}, expires_delta=access_token_expires
        )

        safe_redirect_url = request.url_for("content_page")
        if redirect_to:
            decoded_redirect_to = unquote_plus(redirect_to)
            parsed_url = urlparse(decoded_redirect_to)
            if (
                not parsed_url.netloc
                and not parsed_url.scheme
                and parsed_url.path.startswith("/")
            ):
                safe_redirect_url = decoded_redirect_to
            else:
                print(
                    f"登录后重定向检测到不安全的 URL: {redirect_to}，已重定向到默认首页。"
                )

        response = RedirectResponse(
            url=safe_redirect_url,
            status_code=status.HTTP_302_FOUND,
        )
        # 将 JWT 设置为 HTTP Only Cookie
        # httponly=True 防止客户端 JavaScript 访问 cookie，增强安全性
        # secure=True 仅在 HTTPS 连接上发送 cookie，生产环境强烈建议
        # samesite="Lax" 或 "Strict" 限制第三方网站访问 cookie，防止 CSRF
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # max_age 是秒
            expires=datetime.now(timezone.utc)
            + access_token_expires,  # expires 是日期时间对象
            path="/",  # Cookie 对所有路径可用
            secure=True,  # 生产环境请打开此项，因为GitHub Pages是HTTPS
            samesite="Lax",  # 生产环境请根据需求设置
        )
        return response
    else:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error_message": "验证码错误或已过期。",
                "redirect_to": redirect_to,
            },
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


@app.get("/", response_class=HTMLResponse, name="content_page")
async def content_page(
    request: Request, user_email: str = Depends(get_current_user_jwt)
):
    return templates.TemplateResponse(
        "index.html", {"request": request, "user_email": user_email}
    )


@app.get("/video_feed")
async def video_feed(user_email: str = Depends(get_current_user_jwt)):
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/upload_image")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    user_email: str = Depends(get_current_user_jwt),
):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="不支持的文件类型")

    try:
        file_extension = file.filename.rsplit(".", 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_extension}"

        original_filename_on_disk = f"original_{unique_filename}"
        upload_path = os.path.join(UPLOAD_FOLDER, original_filename_on_disk)

        contents = await file.read()
        await asyncio.to_thread(lambda: open(upload_path, "wb").write(contents))

        processed_image_np = await asyncio.to_thread(
            process_image_to_grayscale, upload_path
        )

        if processed_image_np is not None:
            processed_filename_on_disk = f"grayscale_{unique_filename}"
            processed_path = os.path.join(PROCESSED_FOLDER, processed_filename_on_disk)

            await asyncio.to_thread(cv2.imwrite, processed_path, processed_image_np)

            return {
                "original_image_url": request.url_for(
                    "uploaded_image", filename=original_filename_on_disk
                ),
                "processed_image_url": request.url_for(
                    "processed_image", filename=processed_filename_on_disk
                ),
            }
        else:
            raise HTTPException(
                status_code=500, detail="图片处理失败，请检查图片内容。"
            )

    except Exception as e:
        print(f"文件上传或处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传或处理失败: {e}")


@app.get("/uploaded_images/{filename}")
async def uploaded_image(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="原始图片未找到")
    _, ext = os.path.splitext(filename)
    media_type = f"image/{ext[1:].lower()}" if ext else "image/jpeg"
    return FileResponse(file_path, media_type=media_type)


@app.get("/processed_images/{filename}")
async def processed_image(filename: str):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="处理后的图片未找到")
    _, ext = os.path.splitext(filename)
    media_type = f"image/{ext[1:].lower()}" if ext else "image/jpeg"
    return FileResponse(file_path, media_type=media_type)


@app.get("/logout")
async def logout(request: Request):
    response = RedirectResponse(
        url=request.url_for("login_page"), status_code=status.HTTP_302_FOUND
    )
    # 移除 JWT Cookie
    response.delete_cookie("access_token")
    return response


# --- 性能测试 (保持不变) ---
if __name__ == "__main__":
    import uvicorn
    import threading
    import webbrowser

    test_image_path = "test.png"
    if os.path.exists(test_image_path):
        try:
            import numpy as np
            import cv2 as cv

            img_for_test = cv.imread(test_image_path)
            if img_for_test is not None:
                print("\n--- 灰度转换性能测试 ---")

                def numpy_for_gray_value_test():
                    width, height, channel = img_for_test.shape

                    def get_gray_value(r, g, b):
                        gray_value = 0.299 * r + 0.587 * g + 0.114 * b
                        return gray_value

                    gray_result = np.zeros((width, height))
                    for i in range(width):
                        for j in range(height):
                            gray_result[i, j] = get_gray_value(
                                img_for_test[i, j, 2],
                                img_for_test[i, j, 1],
                                img_for_test[i, j, 0],
                            )
                    return gray_result

                def numpy_gray_value_test():
                    r = img_for_test[:, :, 2]
                    g = img_for_test[:, :, 1]
                    b = img_for_test[:, :, 0]
                    gray_value = 0.299 * r + 0.587 * g + 0.114 * b
                    return gray_value

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
                print(
                    f"警告: 无法读取用于性能测试的图片: '{test_image_path}'. 请确保文件存在且可读。"
                )
        except Exception as e:
            print(f"性能测试过程中发生错误: {e}")
    else:
        print(f"警告: 用于性能测试的图片 '{test_image_path}' 不存在。跳过性能测试。")

    print("启动服务器...")
    print("登录页面地址: http://localhost:8000/login")
    print("视频流地址: http://localhost:8000/video_feed (需要登录)")
    print("图片处理页面: http://localhost:8000/ (需要登录)")

    print(
        f"人脸分类器路径: {cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'}"
    )
    print(f"眼睛分类器路径: {cv2.data.haarcascades + 'haarcascade_eye.xml'}")

    threading.Timer(1, lambda: webbrowser.open("http://localhost:8000/login")).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)

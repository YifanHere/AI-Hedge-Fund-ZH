from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio

from app.backend.routes import api_router
from app.backend.database.connection import engine
from app.backend.database.models import Base
from app.backend.services.ollama_service import ollama_service

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Hedge Fund API", description="Backend API for AI Hedge Fund", version="0.1.0")

# 初始化数据库表（多次运行是安全的）
Base.metadata.create_all(bind=engine)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # 前端URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含所有路由
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    """启动事件以检查Ollama可用性。"""
    try:
        logger.info("检查Ollama可用性...")
        status = await ollama_service.check_ollama_status()
        
        if status["installed"]:
            if status["running"]:
                logger.info(f"✓ Ollama已安装并在{status['server_url']}运行")
                if status["available_models"]:
                    logger.info(f"✓ 可用模型：{', '.join(status['available_models'])}")
                else:
                    logger.info("ℹ 当前没有下载的模型")
            else:
                logger.info("ℹ Ollama已安装但未运行")
                logger.info("ℹ 您可以从设置页面启动它或手动使用'ollama serve'")
        else:
            logger.info("ℹ Ollama未安装。安装它以使用本地模型。")
            logger.info("ℹ 访问 https://ollama.com 下载并安装Ollama")
            
    except Exception as e:
        logger.warning(f"无法检查Ollama状态：{e}")
        logger.info("ℹ 如果您稍后安装Ollama，集成功能将可用")

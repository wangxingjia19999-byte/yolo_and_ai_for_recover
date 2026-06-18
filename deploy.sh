#!/bin/bash
# ══════════════════════════════════════════════════════════════
# 康复训练姿态评估系统 - Mac 本地部署脚本
# ══════════════════════════════════════════════════════════════
#
# 使用方法:
#   chmod +x deploy.sh
#   ./deploy.sh          # 首次部署
#   ./deploy.sh --rebuild # 重新构建镜像
#   ./deploy.sh --stop    # 停止所有服务
#   ./deploy.sh --logs    # 查看日志
# ══════════════════════════════════════════════════════════════

set -e

# 检测 Docker Compose 命令
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker-compose"  # fallback, 会在 check_env 中报错
fi

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════╗"
    echo "║    康复训练姿态评估系统 - Docker 部署工具            ║"
    echo "║    Platform: Mac (Docker Desktop)                    ║"
    echo "╚═══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_env() {
    echo -e "${YELLOW}[1/5] 检查环境...${NC}"

    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker 未安装，请先安装 Docker${NC}"
        exit 1
    fi
    echo -e "  ✅ Docker: $(docker --version)"

    # 检查 Docker Compose (支持 plugin 和独立版本)
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
        echo -e "  ✅ Docker Compose (plugin): $(docker compose version --short)"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
        echo -e "  ✅ Docker Compose: $(docker-compose --version)"
    else
        echo -e "${RED}❌ Docker Compose 未安装${NC}"
        echo -e "${YELLOW}  请安装 Docker Desktop: https://www.docker.com/products/docker-desktop/${NC}"
        exit 1
    fi

    echo ""
}

check_env_file() {
    echo -e "${YELLOW}[2/5] 检查配置文件...${NC}"

    if [ ! -f .env ]; then
        if [ -f .env.docker ]; then
            cp .env.docker .env
            echo -e "  📝 已从 .env.docker 创建 .env"
            echo -e "${YELLOW}  ⚠️  请编辑 .env 文件填入实际的 API Key 等配置${NC}"
            echo -e "${YELLOW}     vim .env${NC}"
        else
            echo -e "${RED}❌ 未找到 .env 文件${NC}"
            exit 1
        fi
    else
        echo -e "  ✅ .env 文件已存在"
    fi

    # 检查模型文件
    if [ ! -f yolo11n-pose.pt ]; then
        echo -e "${RED}❌ 未找到 yolo11n-pose.pt 模型文件${NC}"
        echo -e "${YELLOW}  请下载模型: https://github.com/ultralytics/releases${NC}"
        exit 1
    fi
    echo -e "  ✅ 模型文件: yolo11n-pose.pt ($(du -h yolo11n-pose.pt | cut -f1))"

    echo ""
}

build_images() {
    echo -e "${YELLOW}[3/5] 构建 Docker 镜像...${NC}"
    echo -e "${BLUE}  首次构建约需 10-30 分钟，请耐心等待${NC}"
    echo ""

    $COMPOSE_CMD build

    echo ""
    echo -e "  ✅ 镜像构建完成"
    echo ""
}

start_services() {
    echo -e "${YELLOW}[4/5] 启动服务...${NC}"

    # 先启动基础设施
    echo -e "  启动基础设施 (Kafka, PostgreSQL)..."
    $COMPOSE_CMD up -d zookeeper kafka postgres

    # 等待 Kafka 就绪
    echo -e "  等待 Kafka 就绪..."
    sleep 10

    # 启动业务服务
    echo -e "  启动业务服务..."
    $COMPOSE_CMD up -d

    echo ""
    echo -e "  ✅ 所有服务已启动"
    echo ""
}

show_status() {
    echo -e "${YELLOW}[5/5] 服务状态${NC}"
    echo ""

    $COMPOSE_CMD ps

    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✅ 部署完成！${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  🌐 Web 前端:    http://localhost:8501"
    echo ""
    echo -e "  ${BLUE}常用命令:${NC}"
    echo -e "    查看日志:     $COMPOSE_CMD logs -f"
    echo -e "    查看某服务:   $COMPOSE_CMD logs -f yolo-inference"
    echo -e "    重启服务:     $COMPOSE_CMD restart"
    echo -e "    停止服务:     $COMPOSE_CMD down"
    echo -e "    重建镜像:     ./deploy.sh --rebuild"
    echo ""
}

# ── 主流程 ──────────────────────────────────────────────────

print_header

case "${1}" in
    --stop)
        echo -e "${YELLOW}停止所有服务...${NC}"
        $COMPOSE_CMD down
        echo -e "${GREEN}✅ 所有服务已停止${NC}"
        ;;
    --logs)
        $COMPOSE_CMD logs -f ${2:-}
        ;;
    --rebuild)
        check_env
        check_env_file
        echo -e "${YELLOW}停止旧服务...${NC}"
        $COMPOSE_CMD down
        build_images
        start_services
        show_status
        ;;
    --status)
        $COMPOSE_CMD ps
        ;;
    *)
        check_env
        check_env_file
        build_images
        start_services
        show_status
        ;;
esac

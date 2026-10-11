// ==========================================
// PalmSecureAI 摄像头控制脚本
// ==========================================

class CameraHandler {
    constructor() {
        this.stream = null;
        this.videoElement = null;
        this.canvasElement = null;
        this.constraints = {
            video: {
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        };
        this.isRunning = false;
    }

    /**
     * 初始化摄像头
     * @param {HTMLVideoElement} videoElement - 视频元素
     * @param {HTMLCanvasElement} canvasElement - 画布元素（可选）
     * @param {Object} constraints - 摄像头约束（可选）
     */
    async init(videoElement, canvasElement = null, constraints = null) {
        this.videoElement = videoElement;
        this.canvasElement = canvasElement;
        if (constraints) {
            this.constraints = { ...this.constraints, ...constraints };
        }
    }

    /**
     * 启动摄像头
     */
    async start() {
        if (this.isRunning) {
            console.warn('摄像头已经在运行中');
            return;
        }

        try {
            this.stream = await navigator.mediaDevices.getUserMedia(this.constraints);
            this.videoElement.srcObject = this.stream;
            this.isRunning = true;
            return true;
        } catch (error) {
            console.error('启动摄像头失败:', error);
            throw new CameraError('CAMERA_START_FAILED', '无法访问摄像头，请检查权限设置');
        }
    }

    /**
     * 停止摄像头
     */
    stop() {
        if (!this.stream) return;

        this.stream.getTracks().forEach(track => {
            track.stop();
        });
        this.stream = null;
        this.isRunning = false;

        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }
    }

    /**
     * 拍摄照片
     * @param {Object} options - 拍照选项
     * @returns {string} base64 编码的图像
     */
    capture(options = {}) {
        if (!this.isRunning || !this.videoElement) {
            throw new CameraError('CAMERA_NOT_RUNNING', '摄像头未启动');
        }

        const canvas = this.canvasElement || document.createElement('canvas');
        const video = this.videoElement;

        const width = options.width || video.videoWidth;
        const height = options.height || video.videoHeight;

        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext('2d');

        // 翻转图像（前置摄像头）
        if (options.flip !== false && this.constraints.video.facingMode === 'user') {
            ctx.translate(width, 0);
            ctx.scale(-1, 1);
        }

        ctx.drawImage(video, 0, 0, width, height);

        // 应用滤镜
        if (options.filter) {
            this.applyFilter(ctx, options.filter, width, height);
        }

        // 转换为 base64
        const quality = options.quality || 0.9;
        return canvas.toDataURL('image/jpeg', quality);
    }

    /**
     * 获取视频流
     */
    getStream() {
        return this.stream;
    }

    /**
     * 检查摄像头是否在运行
     */
    isActive() {
        return this.isRunning;
    }

    /**
     * 获取可用摄像头列表
     */
    static async getCameraDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(device => device.kind === 'videoinput');
        } catch (error) {
            console.error('获取摄像头设备失败:', error);
            return [];
        }
    }

    /**
     * 检查摄像头权限
     */
    static async checkPermission() {
        try {
            if ('permissions' in navigator) {
                const permission = await navigator.permissions.query({ name: 'camera' });
                return permission.state;
            }
            return 'prompt';
        } catch (error) {
            return 'unknown';
        }
    }

    /**
     * 请求摄像头权限
     */
    static async requestPermission() {
        try {
            await navigator.mediaDevices.getUserMedia({ video: true });
            return true;
        } catch (error) {
            return false;
        }
    }

    /**
     * 应用滤镜
     */
    applyFilter(ctx, filter, width, height) {
        const imageData = ctx.getImageData(0, 0, width, height);
        const data = imageData.data;

        switch (filter) {
            case 'grayscale':
                for (let i = 0; i < data.length; i += 4) {
                    const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
                    data[i] = avg;
                    data[i + 1] = avg;
                    data[i + 2] = avg;
                }
                break;

            case 'brightness':
                const value = 20;
                for (let i = 0; i < data.length; i += 4) {
                    data[i] = Math.min(255, data[i] + value);
                    data[i + 1] = Math.min(255, data[i + 1] + value);
                    data[i + 2] = Math.min(255, data[i + 2] + value);
                }
                break;

            case 'contrast':
                const factor = (259 * (128 + 255)) / (255 * (259 - 128));
                for (let i = 0; i < data.length; i += 4) {
                    data[i] = Math.min(255, Math.max(0, factor * (data[i] - 128) + 128));
                    data[i + 1] = Math.min(255, Math.max(0, factor * (data[i + 1] - 128) + 128));
                    data[i + 2] = Math.min(255, Math.max(0, factor * (data[i + 2] - 128) + 128));
                }
                break;
        }

        ctx.putImageData(imageData, 0, 0);
    }
}

// 摄像头错误类
class CameraError extends Error {
    constructor(code, message) {
        super(message);
        this.name = 'CameraError';
        this.code = code;
    }
}

// 创建全局摄像头实例
const camera = new CameraHandler();

// 导出
window.CameraHandler = CameraHandler;
window.CameraError = CameraError;
window.camera = camera;

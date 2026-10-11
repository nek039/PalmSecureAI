// ==========================================
// PalmSecureAI Web 应用通用脚本
// ==========================================

// API 基础地址
const API_BASE = '/api';

// 显示加载遮罩
function showLoading(text = '处理中...') {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.querySelector('.loading-text').textContent = text;
        overlay.style.display = 'flex';
    }
}

// 隐藏加载遮罩
function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// 显示提示消息
function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${getToastIcon(type)}</span>
        <span class="toast-message">${escapeHtml(message)}</span>
    `;

    container.appendChild(toast);

    // 自动移除
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// 获取提示图标
function getToastIcon(type) {
    const icons = {
        success: '✓',
        error: '✗',
        warning: '⚠',
        info: 'ℹ'
    };
    return icons[type] || icons.info;
}

// HTML 转义
function escapeHtml(text) {
    if (typeof text !== 'string') return text;
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 格式化日期时间
function formatDateTime(dateStr) {
    if (!dateStr) return '--';
    const date = new Date(dateStr);
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// 格式化时间
function formatTime(dateStr) {
    if (!dateStr) return '--';
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now - date;

    if (diff < 60000) return '刚刚';
    if (diff < 3600000) return Math.floor(diff / 60000) + '分钟前';
    if (diff < 86400000) return Math.floor(diff / 3600000) + '小时前';
    return date.toLocaleDateString('zh-CN');
}

// 格式化相对时间
function formatRelativeTime(dateStr) {
    if (!dateStr) return '--';
    const date = new Date(dateStr);
    return date.toLocaleString('zh-CN');
}

// 将文件转换为 base64
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// 压缩图像
async function compressImage(dataUrl, maxWidth = 640, quality = 0.9) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            let width = img.width;
            let height = img.height;

            // 按比例缩放
            if (width > maxWidth) {
                height = (height * maxWidth) / width;
                width = maxWidth;
            }

            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, width, height);

            resolve(canvas.toDataURL('image/jpeg', quality));
        };
        img.src = dataUrl;
    });
}

// API 请求封装
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;

    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json'
        }
    };

    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers
        }
    };

    try {
        const response = await fetch(url, mergedOptions);
        const data = await response.json();
        return { response, data };
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// 获取系统状态
async function getSystemStatus() {
    try {
        const { data } = await apiRequest('/status');
        return data.success ? data.data : null;
    } catch (error) {
        console.error('获取系统状态失败:', error);
        return null;
    }
}

// 检查系统健康
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        return response.ok;
    } catch (error) {
        return false;
    }
}

// 更新系统状态指示器
async function updateSystemStatus() {
    const statusEl = document.getElementById('systemStatus');
    if (!statusEl) return;

    const isHealthy = await checkHealth();
    const dot = statusEl.querySelector('.status-dot');
    const text = statusEl.querySelector('.status-text');

    if (isHealthy) {
        dot.style.background = 'var(--success-color)';
        text.textContent = '系统在线';
    } else {
        dot.style.background = 'var(--danger-color)';
        text.textContent = '系统离线';
    }
}

// 定时更新系统状态
setInterval(updateSystemStatus, 30000);
updateSystemStatus();

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    // 添加全局点击事件处理
    document.addEventListener('click', function(e) {
        // 处理模态框关闭
        if (e.target.classList.contains('modal-overlay')) {
            const modal = e.target.closest('.modal');
            if (modal) {
                modal.style.display = 'none';
            }
        }
    });

    // 处理 ESC 键关闭模态框
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            document.querySelectorAll('.modal').forEach(modal => {
                modal.style.display = 'none';
            });
        }
    });

    // 表单提交处理（排除注册页面）
    document.querySelectorAll('form').forEach(form => {
        // 跳过注册页面，避免干扰按钮点击
        if (form.closest('.enroll-page')) {
            return;
        }

        form.addEventListener('submit', function(e) {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn && !submitBtn.disabled) {
                submitBtn.disabled = true;
                setTimeout(() => {
                    submitBtn.disabled = false;
                }, 2000);
            }
        });
    });

    // 初始化所有选择器
    initSelects();

    // 初始化所有日期时间输入
    initDateTimeInputs();
});

// 初始化选择器
function initSelects() {
    document.querySelectorAll('select').forEach(select => {
        select.addEventListener('change', function() {
            this.dispatchEvent(new CustomEvent('select:changed', {
                detail: { value: this.value }
            }));
        });
    });
}

// 初始化日期时间输入
function initDateTimeInputs() {
    document.querySelectorAll('input[type="datetime-local"]').forEach(input => {
        const now = new Date();
        const year = now.getFullYear();
        const month = String(now.getMonth() + 1).padStart(2, '0');
        const day = String(now.getDate()).padStart(2, '0');
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');

        input.value = `${year}-${month}-${day}T${hours}:${minutes}`;
    });
}

// 防抖函数
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// 节流函数
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// 复制到剪贴板
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showToast('已复制到剪贴板', 'success');
    } catch (error) {
        console.error('复制失败:', error);
        showToast('复制失败', 'error');
    }
}

// 下载文件
function downloadFile(data, filename, type = 'application/json') {
    const blob = new Blob([data], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}

// 导出
window.PalmSecureAPI = {
    showLoading,
    hideLoading,
    showToast,
    escapeHtml,
    formatDateTime,
    formatTime,
    formatRelativeTime,
    fileToBase64,
    compressImage,
    apiRequest,
    getSystemStatus,
    checkHealth,
    updateSystemStatus,
    copyToClipboard,
    downloadFile,
    debounce,
    throttle
};

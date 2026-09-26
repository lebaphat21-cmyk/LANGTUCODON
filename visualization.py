"""Canvas chỉ hiển thị lịch sử đã tính, không thay đổi khoảng cách dữ liệu."""
from html import escape
import json
from pathlib import Path


def replay_payload(X, model):
    if len(model.sample_indices_) != len(X):
        raise ValueError('Canvas phát lại toàn bộ dữ liệu đầu vào; không nhận mô hình lấy mẫu con.')
    return {'points': X.tolist(), 'history': model.history_, 'k': model.n_clusters}


def render_canvas_html(X, model, title='Mô phỏng CURE'):
    template = (Path(__file__).parent / 'templates' / 'cure_replay.html').read_text(encoding='utf-8-sig')
    payload = json.dumps(replay_payload(X, model), ensure_ascii=False).replace('<', '\\u003c')
    return template.replace('__TITLE__', escape(title)).replace('__PAYLOAD__', payload)

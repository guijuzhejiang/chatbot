import os

from sanic import Blueprint
from sanic import Sanic

from handlers.asr import Asr
from sanic_demo.lib.common.AsyncRpcClient_v2 import AsyncRpcClient
from sanic_demo.utils.config import CONFIG


app = Sanic(__name__)

bp = Blueprint("FER")
bp.static('/static', os.path.join(os.path.dirname(__file__), 'static'))
bp.add_route(Asr.as_view(), "/")


@bp.listener("before_server_start")
async def before_server_start(app, loop):
    app.ctx.pika_rpc = await AsyncRpcClient(loop).connect()

app.blueprint(bp)

# app.add_route(FaceEmotionRec.as_view(), "/face_emotion_rec")


if __name__ == "__main__":
    if CONFIG['server']['ssl']:
        ssl_config = {"cert": CONFIG['server']['cert'], "key": CONFIG['server']['key']}
    else:
        ssl_config = None
    app.run(host=CONFIG['server']['host'],
            port=CONFIG['server']['port'],
            workers=CONFIG['server']['workers'],
            access_log=CONFIG['server']['access_log'],
            debug=CONFIG['server']['debug'],
            ssl=ssl_config)

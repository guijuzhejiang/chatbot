import json
import base64

from sanic.response import text
from sanic.views import HTTPMethodView

from sanic_demo.lib.common.sanic_jinja2 import SanicJinja2
from sanic_demo.utils.config import CONFIG


class Asr(HTTPMethodView):
    async def post(self, request):
        form_data = {k: v[0] for k, v in dict(request.form).items()}
        if not 'b64' in form_data.keys() or form_data['b64'] == None or form_data['b64'] == "":
            form_data['b64'] = base64.b64encode(request.files["wav_file"][0].body).decode()
        return text(await request.app.ctx.pika_rpc.call(json.dumps(form_data), target_mq_name=CONFIG['asr_mq_name']))

    async def get(self, request):
        return await SanicJinja2.template_render_async("index.html")
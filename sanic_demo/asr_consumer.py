# -*- coding:utf-8 -*-
import asyncio
import base64
import datetime
import json
import os
import traceback

# import soundfile
from aio_pika import connect_robust
from aio_pika.patterns import RPC

from lib.common.common_util import logging
from sanic_demo.utils.config import WORK_DIR, CONFIG

mq_name = CONFIG['asr_mq_name']
start_time = datetime.datetime.now()

if __name__ == "__main__":
    context = []

    async def on_request(*, json_msg):
        try:
            tp = datetime.datetime.now()
            form_data = json.loads(json_msg)
            # print(form_data)
            wav_fp = os.path.join(WORK_DIR, f'data/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.wav')
            bin_wav = base64.b64decode(form_data['b64'])
            fout = open(wav_fp, 'wb')
            fout.write(bin_wav)
            fout.close()

            # wav, rate = soundfile.read(wav_fp)
            res_text = transcribe(wav_fp)

            print(f"asr process time: {datetime.datetime.now()-tp}")
            return json.dumps({'success': True, 'res_text': res_text})

        except Exception:
            response = ''
            print(traceback.format_exc())
            logging(
                f"[{__file__}][{datetime.datetime.now()}]:"
                f"{traceback.format_exc()}",
                os.path.join(WORK_DIR, 'logs/error.log'))

            return json.dumps({'success': False, 'err': traceback.format_exc()})


    async def main():
        connection = await connect_robust(
            "amqp://guest:guest@127.0.0.1/",
            client_properties={"connection_name": "callee"},
        )

        # Creating channel
        channel = await connection.channel()

        rpc = await RPC.create(channel)
        await rpc.register(mq_name, on_request, auto_delete=True)
        print(" [x] Awaiting RPC requests")
        return connection


    # pika
    loop = asyncio.get_event_loop()
    connection = loop.run_until_complete(main())

    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(connection.close())
        loop.shutdown_asyncgens()

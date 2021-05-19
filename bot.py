import torch, torchvision
import numpy as np
import cv2
import io
import os
from PIL import Image
import aiohttp
import asyncio
import logging
from aiogram import Bot, Dispatcher, executor, types
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer
from name_clas import boxes
from nick import sav, writ

NON_TARGET_CONTENT_TYPES = [
'text','audio', 'document', 'sticker','video', 'video_note', 'voice','location',
'contact', 'new_chat_members','left_chat_member','new_chat_title', 'new_chat_photo',
'delete_chat_photo','group_chat_created', 'supergroup_chat_created', 'channel_chat_created',
'migrate_to_chat_id', 'migrate_from_chat_id', 'pinned_message'
]

HELLO_TEXT = """Привет, %s! Пришли мне, пожалуйста, фото ладони как на фото ниже в примере и я постораюсь
тебя удивить. Для более качественной работы отправляй фото точно как в примере."""
NON_TARGET_TEXT = "Прости, %s, я могу работать только с фото!"
WAITING_TEXT = "Идет обработка фотографии это может занять некоторе время."

#Register custom COCO dataset
register_coco_instances("palm", {}, "Palm_coco_detection.json", "./palm/")
clothesnet = MetadataCatalog.get("palm")
dataset_dicts = DatasetCatalog.get("palm")
                       #COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml  ---- 002
                       #COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml ------- 004
                       #COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --- 003
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.27  # set threshold for this model
cfg.MODEL.WEIGHTS = ("model_002.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

#Configure logging
logging.basicConfig(level=logging.INFO)

TOKEN = os.getenv('TOKENBOT')
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

with open('photo.jpg', 'rb') as target:
    photo_start = target.read()

#Base command for start bot
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    chat_id = message.chat.id
    user_name = message.from_user.first_name
    user_id = message.from_user.id
    text = HELLO_TEXT %user_name
    logging.info(f'First start from user_name = {user_name}, user_id = {user_id}')
    await message.reply(text)
    await bot.send_photo(chat_id, photo=photo_start)

@dp.message_handler(content_types=NON_TARGET_CONTENT_TYPES)
async def handle_docs_photo(message):
    user_name = message.from_user.first_name
    text = NON_TARGET_TEXT %user_name
    await message.reply(text)

@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(message):
    chat_id = message.chat.id

    if message.media_group_id is None:
        # Get user's variables
        user_name = message.from_user.first_name
        user_id = message.from_user.id
        message_id = message.message_id
        text_user =  writ(user_name) # чтение файла с никами
        if text_user == 'Что-то новенькое!':
            sav(user_name) # запись ника в файл если такоего нету.
        text = f"{text_user}\n{WAITING_TEXT}"
        logging.info(f'{user_name, user_id} is knocking to our bot')
        await bot.send_message(chat_id, text)

        photo_name = './input/photo_%s_%s.jpg' %(user_name, message_id)
        await message.photo[-1].download(photo_name) # extract photo for further procceses

        #Photo processing
        photo_output, text = predict(photo_name)
        await bot.send_photo(chat_id, photo_output)
        await bot.send_message(chat_id, text)

    else:
        text = NOT_TARGET_TEXT %user_name
        await message.reply(text)

def predict(photo_name):
    im = cv2.imread(photo_name)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=clothesnet,
                   scale=0.8,
    )


    outputs["instances"].remove('pred_boxes')
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    is_success, buffer = cv2.imencode(".jpg", v.get_image()[:, :, ::-1])
    #print(buffer)
    bio = io.BytesIO(buffer)
    bio.name = 'image.jpeg'
    bio.seek(0)
    text = sorted(set(np.array(outputs["instances"].pred_classes)), reverse=True)

    return bio, boxes(text)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

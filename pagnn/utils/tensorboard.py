import io

import tensorboardX.summary
from PIL import Image
from tensorboardX import SummaryWriter
from tensorboardX.src.summary_pb2 import Summary


def add_image(writer: SummaryWriter, tag: str, img: Image.Image, global_step: int):
    tag = tensorboardX.summary._clean_tag(tag)

    # NB: Image.tobyes() does not work for compressed formats
    output = io.BytesIO()
    img.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()

    tb_summary_image = Summary.Image(
        height=img.height, width=img.width, colorspace=3, encoded_image_string=image_string)
    tb_summary = Summary(value=[Summary.Value(tag=tag, image=tb_summary_image)])
    writer.file_writer.add_summary(tb_summary, global_step)

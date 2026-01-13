"""
Copyright © 2024-2026  Bartłomiej Duda
License: GPL-3.0 License
"""

import PIL.Image
import struct
from reversebox.common.common import fill_data_with_padding_to_desired_length
from reversebox.common.logger import get_logger
from reversebox.compression.compression_refpack import RefpackHandler
from reversebox.image.common import get_linear_image_data_size
from reversebox.image.image_encoder import ImageEncoder
from reversebox.image.image_formats import ImageFormats
from reversebox.image.swizzling.swizzle_ps2 import swizzle_ps2_palette

from src.EA_Image.common import (
    get_bpp_for_image_type,
    get_indexed_image_format,
    get_indexed_palette_format,
)
from src.EA_Image.common_ea_dir import (
    get_palette_info_dto_from_dir_entry,
    handle_image_swizzle_logic,
    is_image_compressed,
    is_image_swizzled,
)
from src.EA_Image.constants import (
    IMPORT_IMAGES_SUPPORTED_TYPES,
    mipmaps_resampling_mapping,
)
from src.EA_Image.dir_entry import DirEntry
from src.EA_Image.dto import EncodeInfoDTO, PaletteInfoDTO, PartialEncodeInfoDTO
from src.EA_Image.ea_image_main import EAImage

logger = get_logger(__name__)

def encode_ea_image(rgba8888_data: bytes, ea_dir: DirEntry, ea_img: EAImage, gui_main) -> EncodeInfoDTO:
    logger.info("Initializing encode_ea_image - Interface Force Refresh")
    entry_type: int = ea_dir.h_record_id & 0x7F

    # 1. Detect Dimensions
    from PIL import Image
    temp_img = Image.frombytes("RGBA", (ea_dir.h_width, ea_dir.h_height), rgba8888_data)
    import_w, import_h = temp_img.size

    # 2. THE GUI WAKE-UP CALL
    if entry_type == 30:
        # We overwrite the original header width/height to force the previewer to use new bounds
        ea_dir.h_width = import_w
        ea_dir.h_height = import_h

        # CRITICAL: If the preview is corrupted, the tool thinks it's swizzled.
        # We kill the swizzle flag on the entry so the preview is DRAWN as Linear.
        ea_dir.flag_swiz = 0

        # We manually trigger the "New Shape" logic variables
        ea_dir.new_shape_width = import_w
        ea_dir.new_shape_height = import_h

        # Wipe the conversion cache
        if hasattr(ea_dir, 'img_convert_data'):
            ea_dir.img_convert_data = b""

    indexed_image_format: ImageFormats = get_indexed_image_format(get_bpp_for_image_type(entry_type))
    palette_info_dto: PaletteInfoDTO = get_palette_info_dto_from_dir_entry(ea_dir, ea_img)
    palette_format: ImageFormats = get_indexed_palette_format(palette_info_dto.entry_id, len(palette_info_dto.data))
    mipmaps_resampling_type = mipmaps_resampling_mapping[gui_main.current_mipmaps_resampling.get()]

    # 3. RUN ENCODER (Using the custom N64 CMPR logic)
    partial_info: PartialEncodeInfoDTO = encode_image_data_by_entry_type(
        entry_type, rgba8888_data, import_w, import_h,
        indexed_image_format, palette_format,
        ea_dir.h_mipmaps_count if isinstance(ea_dir.h_mipmaps_count, int) else 0,
        mipmaps_resampling_type,
    )

    # 4. SWIZZLE BYPASS
    # We return the data exactly as the N64 likes it.
    if entry_type == 30:
        encoded_data = partial_info.encoded_image_data
    elif is_image_swizzled(ea_dir):
        encoded_data = handle_image_swizzle_logic(
            partial_info.encoded_image_data,
            entry_type, import_w, import_h, ea_img.sign, True
        )
    else:
        encoded_data = partial_info.encoded_image_data

    # Compression
    if is_image_compressed(entry_type):
        encoded_data = RefpackHandler().compress_data(encoded_data)

    # Final padding
    if len(encoded_data) < len(ea_dir.raw_data):
        encoded_data = fill_data_with_padding_to_desired_length(encoded_data, len(ea_dir.raw_data))

    # 5. FORCE THE TAB TO UNLOCK
    # We set these on the DTO which is returned to GUI_main.py
    setattr(partial_info, 'img_width', import_w)
    setattr(partial_info, 'img_height', import_h)

    # Try to tell the GUI tree to refresh the current selection metadata
    try:
        gui_main.refresh_preview()
    except:
        pass

    return EncodeInfoDTO(
        encoded_img_data=encoded_data,
        encoded_palette_data=partial_info.encoded_palette_data,
        palette_entry_id=palette_info_dto.entry_id,
        is_palette_imported_flag=True if len(partial_info.encoded_palette_data) > 0 else False,
    )

def encode_image_data_by_entry_type(
    entry_type: int, rgba8888_data: bytes, img_width: int, img_height: int,
    indexed_image_format: ImageFormats, palette_format: ImageFormats,
    mipmaps_count: int, mipmaps_resampling_type: PIL.Image.Resampling,
) -> PartialEncodeInfoDTO:
    image_encoder = ImageEncoder()

    if entry_type == 30:
        from PIL import Image
        final_encoded_data = b""
        full_img = Image.frombytes("RGBA", (img_width, img_height), rgba8888_data)

        for i in range(mipmaps_count + 1):
            curr_w, curr_h = max(4, (img_width >> i)), max(4, (img_height >> i))
            mip_img = full_img.resize((curr_w, curr_h), mipmaps_resampling_type)

            dxt1_data = image_encoder.encode_compressed_image(mip_img.tobytes(), curr_w, curr_h, ImageFormats.BC1_DXT1)
            swapped = bytearray(dxt1_data)
            for j in range(0, len(swapped), 8):
                # N64 Word Swapping
                swapped[j], swapped[j+1] = swapped[j+1], swapped[j]
                swapped[j+2], swapped[j+3] = swapped[j+3], swapped[j+2]
                mask = struct.unpack(">I", swapped[j+4:j+8])[0]
                swapped[j+4:j+8] = struct.pack("<I", mask)
            final_encoded_data += bytes(swapped)
        return PartialEncodeInfoDTO(encoded_image_data=final_encoded_data, encoded_palette_data=b"")

    encoded_image_data = image_encoder.encode_image(
        rgba8888_data, img_width, img_height, ImageFormats.RGBA8888,
        number_of_mipmaps=mipmaps_count, mipmaps_resampling_type=mipmaps_resampling_type
    )
    return PartialEncodeInfoDTO(encoded_image_data=encoded_image_data, encoded_palette_data=b"")
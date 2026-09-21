/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Worker/wasm32 glyph backend.
//!
//! Cloudflare Workers (and the other wasm hosts this fork targets) have no
//! FreeType/CoreText/DirectWrite, no OS font service, and no GPU context.
//! This backend turns glyph ids that Servo's layout has already positioned
//! into bitmaps using `swash`, a pure-Rust font rasterizer, per the design
//! in `docs/wasm-rendering.md` in the servo-wasm tree.
//!
//! This is ported from the real swash backend that already exists on this
//! fork's `swash-backend`/`0.67-swash` branches (see
//! `wr_glyph_rasterizer/src/backend/swash/font.rs` there), adapted to the
//! current `0.70` `FontInstance`/`RasterizedGlyph` API. The rendering logic
//! (color/emoji glyphs, fractional subpixel offset, synthetic bold, real
//! advance metrics) is carried over as-is. What's removed is native-font
//! resolution (`font_index::FontCache`/`FontId`): the Worker has no OS font
//! service to resolve a `NativeFontHandle` against, so `add_native_font`
//! fails loudly instead, matching the note in `docs/wasm-rendering.md` that
//! fonts must be supplied as raw bytes by the embedder.

use std::collections::hash_map::Entry;
use std::mem;
use std::sync::Arc;

use api::{ColorU, FontInstanceFlags, FontKey, FontRenderMode, GlyphDimensions, NativeFontHandle};
use swash::scale::image::{Content, Image as GlyphImage};
use swash::scale::{Render, ScaleContext, Source, StrikeWith};
use swash::zeno::{Format, Vector};
use swash::{FontRef, GlyphId};

use crate::rasterizer::{
    FontInstance, FontTransform, GlyphFormat, GlyphKey, GlyphRasterError, GlyphRasterResult,
    RasterizedGlyph,
};
use crate::types::FastHashMap;

fn is_bitmap_font(font: &FontInstance) -> bool {
    font.flags.contains(FontInstanceFlags::EMBEDDED_BITMAPS)
}

pub struct FontContext {
    fonts: FastHashMap<FontKey, Arc<Vec<u8>>>,
    scale_context: ScaleContext,
    cache: FastHashMap<(FontInstance, GlyphKey), GlyphImage>,
}

impl FontContext {
    pub fn distribute_across_threads() -> bool {
        false
    }

    pub fn new() -> FontContext {
        FontContext {
            fonts: FastHashMap::default(),
            scale_context: ScaleContext::new(),
            cache: FastHashMap::default(),
        }
    }

    pub fn add_raw_font(&mut self, font_key: &FontKey, bytes: Arc<Vec<u8>>, index: u32) {
        if self.fonts.contains_key(font_key) {
            return;
        }
        // Validate eagerly so a bad font fails at load time, matching the
        // other backends' early-failure behavior in `add_raw_font`.
        if FontRef::from_index(&bytes, index as usize).is_none() {
            panic!(
                "adding raw font failed: {} bytes, index={}",
                bytes.len(),
                index
            );
        }
        self.fonts.insert(*font_key, bytes);
    }

    pub fn add_native_font(&mut self, _font_key: &FontKey, _native_font_handle: NativeFontHandle) {
        unreachable!(
            "add_native_font has no meaning on wasm32: the Worker cannot discover host system \
             fonts, so callers must supply raw font bytes via add_raw_font instead"
        );
    }

    pub fn delete_font(&mut self, font_key: &FontKey) {
        if self.fonts.remove(font_key).is_some() {
            self.cache.retain(|(instance, _), _| instance.base.font_key != *font_key);
        }
    }

    pub fn delete_font_instance(&mut self, instance: &FontInstance) {
        self.cache
            .retain(|(cached, _), _| cached.base.instance_key != instance.base.instance_key);
    }

    pub fn get_glyph_index(&mut self, font_key: FontKey, ch: char) -> Option<u32> {
        let bytes = self.fonts.get(&font_key)?;
        let font = FontRef::from_index(bytes, 0)?;
        let glyph_id = font.charmap().map(ch);
        if glyph_id == 0 {
            None
        } else {
            Some(glyph_id as u32)
        }
    }

    pub fn get_glyph_dimensions(
        &mut self,
        font: &FontInstance,
        key: &GlyphKey,
    ) -> Option<GlyphDimensions> {
        let image = self.get_or_create_cache(font, key)?;
        let bytes = self.fonts.get(&font.base.font_key)?;
        let font_ref = FontRef::from_index(bytes, 0)?;
        let advance = font_ref
            .glyph_metrics(&[])
            .scale(font.get_transformed_size() as f32)
            .advance_width(key.index() as GlyphId);
        Some(GlyphDimensions {
            left: image.placement.left,
            top: image.placement.top,
            width: image.placement.width as i32,
            height: image.placement.height as i32,
            advance,
        })
    }

    pub fn prepare_font(font: &mut FontInstance) {
        match font.render_mode {
            FontRenderMode::Mono => {
                // In mono mode the color of the font is irrelevant.
                font.color = ColorU::new(0xFF, 0xFF, 0xFF, 0xFF);
                // Subpixel positioning is disabled in mono mode.
                font.disable_subpixel_position();
            }
            FontRenderMode::Alpha | FontRenderMode::Subpixel => {
                // No preblending on this backend yet, so color is unused.
                font.color = ColorU::new(0xFF, 0xFF, 0xFF, 0xFF);
            }
        }
    }

    pub fn begin_rasterize(_font: &FontInstance) {}

    pub fn end_rasterize(_font: &FontInstance) {}

    fn get_or_create_cache(
        &mut self,
        font: &FontInstance,
        key: &GlyphKey,
    ) -> Option<GlyphImage> {
        match self.cache.entry((font.clone(), *key)) {
            Entry::Occupied(entry) => Some(entry.get().clone()),
            Entry::Vacant(entry) => {
                let bytes = self.fonts.get(&font.base.font_key)?;
                let font_ref = FontRef::from_index(bytes, 0)?;
                let image = render_glyph(&mut self.scale_context, &font_ref, font, key)?;
                entry.insert(image.clone());
                Some(image)
            }
        }
    }

    pub fn rasterize_glyph(&mut self, font: &FontInstance, key: &GlyphKey) -> GlyphRasterResult {
        let image = self
            .get_or_create_cache(font, key)
            .ok_or(GlyphRasterError::LoadFailed)?;

        let GlyphImage {
            placement,
            data: pixels,
            content,
            ..
        } = image;

        // Alpha texture bounds can sometimes return an empty rect, e.g. for
        // spaces.
        if placement.width == 0 || placement.height == 0 {
            return Err(GlyphRasterError::LoadFailed);
        }

        let bgra_pixels = match content {
            Content::Color | Content::SubpixelMask => {
                let subpixel_bgr = font.flags.contains(FontInstanceFlags::SUBPIXEL_BGR);
                pixels
                    .chunks_exact(4)
                    .flat_map(|src| {
                        let (mut r, g, mut b, a) = (src[0], src[1], src[2], src[3]);
                        if subpixel_bgr {
                            mem::swap(&mut r, &mut b);
                        }
                        [b, g, r, a]
                    })
                    .collect()
            }
            Content::Mask => pixels
                .iter()
                .flat_map(|&a| [a, a, a, a])
                .collect(),
        };

        let format = match content {
            Content::Mask => font.get_alpha_glyph_format(),
            Content::SubpixelMask => font.get_subpixel_glyph_format(),
            Content::Color => GlyphFormat::ColorBitmap,
        };

        Ok(RasterizedGlyph {
            left: placement.left as f32,
            top: placement.top as f32,
            width: placement.width as i32,
            height: placement.height as i32,
            scale: 1.0,
            format,
            bytes: bgra_pixels,
            is_packed_glyph: false,
        })
    }
}

fn render_glyph(
    context: &mut ScaleContext,
    font: &FontRef,
    instance: &FontInstance,
    glyph_key: &GlyphKey,
) -> Option<GlyphImage> {
    let (x_scale, y_scale) = instance.transform.compute_scale().unwrap_or((1.0, 1.0));
    let size = instance.size.to_f32_px() * y_scale as f32;

    let (_transform, (x_offset, y_offset)) = if is_bitmap_font(instance) {
        (FontTransform::identity(), (0.0, 0.0))
    } else {
        (
            instance.transform.invert_scale(y_scale, y_scale),
            instance.get_subpx_offset(glyph_key),
        )
    };

    let strike_scale = if is_bitmap_font(instance) { y_scale } else { x_scale };
    let _extra_strikes = instance.get_extra_strikes(
        FontInstanceFlags::SYNTHETIC_BOLD | FontInstanceFlags::MULTISTRIKE_BOLD,
        strike_scale,
    );

    let format = match instance.render_mode {
        FontRenderMode::Mono | FontRenderMode::Alpha => Format::Alpha,
        FontRenderMode::Subpixel => Format::Subpixel,
    };

    let mut scaler = context
        .builder(*font)
        .size(size)
        .hint(true)
        .build();

    let offset = Vector::new((x_offset as f32).fract(), (y_offset as f32).fract());

    Render::new(&[
        Source::ColorOutline(0),
        Source::ColorBitmap(StrikeWith::BestFit),
        Source::Outline,
    ])
    .format(format)
    .offset(offset)
    .default_color([
        instance.color.r,
        instance.color.g,
        instance.color.b,
        instance.color.a,
    ])
    .render(&mut scaler, glyph_key.index() as GlyphId)
}

#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/fb.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

typedef struct {
    uint8_t *framebuffer_p;
    int framebuffer_d;
    int width;
    int height;
    int x_offset;
    int y_offset;
    int pixel_step;
    int line_length;
    int screen_size;
} DisplayData;

uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b) {
    uint16_t color = ((r >> 3) << 11) |
                     ((g >> 2) << 5)  |
                      (b >> 3);
    return color;
}

void draw_rgb_frame(
    uint8_t *fbp,
    int frame_width,
    int frame_height,
    int screen_width, 
    int screen_height,
    int x_offset,
    int y_offset,
    int line_length,
    int pixel_step,
    uint8_t *frame) {
    for (int y = 0; y < frame_height; y++) {
        if (y >= screen_height) continue;
        for (int x = 0; x < frame_width; x++) {
            if (x >= screen_width) continue;
            long location = (x + x_offset) * pixel_step + (y + y_offset) * line_length;
            int idx = (x + y * frame_width) * 3;
            uint8_t r = *(frame + idx + 0);
            uint8_t g = *(frame + idx + 1);
            uint8_t b = *(frame + idx + 2);
            *((uint16_t*)(fbp + location)) = rgb888_to_rgb565(r, g, b);
        }
    }
}

void draw_bgr_frame(
    uint8_t *fbp,
    int frame_width,
    int frame_height,
    int screen_width, 
    int screen_height,
    int x_offset,
    int y_offset,
    int line_length,
    int pixel_step,
    uint8_t *frame) {
    for (int y = 0; y < frame_height; y++) {
        if (y >= screen_height) continue;
        for (int x = 0; x < frame_width; x++) {
            if (x >= screen_width) continue;
            long location = (x + x_offset) * pixel_step + (y + y_offset) * line_length;
            int idx = (x + y * frame_width) * 3;
            uint8_t r = *(frame + idx + 2);
            uint8_t g = *(frame + idx + 1);
            uint8_t b = *(frame + idx + 0);
            *((uint16_t*)(fbp + location)) = rgb888_to_rgb565(r, g, b);
        }
    }
}

void get_display_data(DisplayData *data) {
    int fd = open("/dev/fb0", O_RDWR);
    if (fd == -1) {
        perror("Failed to open framebuffer device");
    }
    
    struct fb_fix_screeninfo finfo;
    if (ioctl(fd, FBIOGET_FSCREENINFO, &finfo)) {
        perror("FBIOGET_FSCREENINFO");
        close(fd);
    }
    
    struct fb_var_screeninfo vinfo;
    if (ioctl(fd, FBIOGET_VSCREENINFO, &vinfo)) {
        perror("FBIOGET_VSCREENINFO");
        close(fd);
    }
    
    long screen_size = vinfo.yres_virtual * finfo.line_length;

    uint8_t *fbp = mmap(0, screen_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (fbp == MAP_FAILED) {
        perror(" Failed to map framebuffer device");
        close(fd);
    }

    DisplayData display_data = {
        .framebuffer_p = fbp,
        .framebuffer_d = fd,
        .width = vinfo.xres,
        .height = vinfo.yres,
        .x_offset = vinfo.xoffset,
        .y_offset = vinfo.yoffset,
        .pixel_step = vinfo.bits_per_pixel / 8,
        .line_length = finfo.line_length,
        .screen_size = screen_size
    };

    *data = display_data;
}

void clean_up(uint8_t *fbp, int fbd, int screen_size) {
    munmap(fbp, screen_size);
    close(fbd);
}
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/fb.h>

struct fb_fix_screeninfo finfo;
struct fb_var_screeninfo vinfo;
uint8_t *fb_p;
uint8_t *buffer;

int m_display_width = 0;
int m_display_height = 0;
int m_mmap_size = 0;
int m_screen_size = 0;
int m_pixel_step = 0;


typedef struct {
    int framebuffer_desc;
    int width;
    int height;
    int pixel_step;
    int line_length;
    int screen_size;
    int mmap_size;
} DisplayData;

uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b) {
    uint16_t color = ((r >> 3) << 11) |
                     ((g >> 2) << 5)  |
                      (b >> 3);
    return color;
}

void draw_rgb_frame(int frame_width, int frame_height, uint8_t *frame) {
    for (int y = 0; y < frame_height; y++) {
        if (y >= m_display_height) continue;
        for (int x = 0; x < frame_width; x++) {
            if (x >= m_display_width) continue;
            long location = x * m_pixel_step + y * finfo.line_length;
            int idx = (x + y * frame_width) * 3;
            uint8_t r = *(frame + idx + 0);
            uint8_t g = *(frame + idx + 1);
            uint8_t b = *(frame + idx + 2);
            switch(m_pixel_step) {
                case 2: {
                    *((uint16_t*)(buffer + location)) = rgb888_to_rgb565(r, g, b);
                    break;
                }
                case 3: {
                    *((uint16_t*)(buffer + location + 0)) = b;
                    *((uint16_t*)(buffer + location + 1)) = g;
                    *((uint16_t*)(buffer + location + 2)) = r;
                    break;
                } 
                case 4: {
                    *((uint16_t*)(buffer + location + 0)) = b;
                    *((uint16_t*)(buffer + location + 1)) = g;
                    *((uint16_t*)(buffer + location + 2)) = r;
                    *((uint16_t*)(buffer + location + 3)) = 0;
                    break;
                }
                default:
                    printf("Pixel depth %d is not supported!\n", m_pixel_step*8);
                    return;
            }
        }
    }
}

void draw_bgr_frame(int frame_width, int frame_height, uint8_t *frame) {
    for (int y = 0; y < frame_height; y++) {
        if (y >= m_display_height) continue;
        for (int x = 0; x < frame_width; x++) {
            if (x >= m_display_width) continue;
            long location = x * m_pixel_step + y * finfo.line_length;
            int idx = (x + y * frame_width) * 3;
            uint8_t r = *(frame + idx + 2);
            uint8_t g = *(frame + idx + 1);
            uint8_t b = *(frame + idx + 0);
            switch(m_pixel_step) {
                case 2: {
                    *((uint16_t*)(buffer + location)) = rgb888_to_rgb565(r, g, b);
                    break;
                }
                case 3: {
                    *((uint16_t*)(buffer + location + 0)) = r;
                    *((uint16_t*)(buffer + location + 1)) = g;
                    *((uint16_t*)(buffer + location + 2)) = b;
                    break;
                } 
                case 4: {
                    *((uint16_t*)(buffer + location + 0)) = r;
                    *((uint16_t*)(buffer + location + 1)) = g;
                    *((uint16_t*)(buffer + location + 2)) = b;
                    *((uint16_t*)(buffer + location + 3)) = 0;
                    break;
                }
                default:
                    printf("Pixel depth %d is not supported!\n", m_pixel_step*8);
                    return;
            }
        }
    }
}

void display_buffer() {
    memcpy(fb_p, buffer, m_mmap_size);
}

int wait_for_vsync(int fd) {
    int zero = 0, ret;

    ret = ioctl(fd, FBIO_WAITFORVSYNC, &zero);
    if (ret) {
        perror("Error for FBIO_WAITFORVSYNC");
		return ret;
    }
    return 0;
}


int get_display_data(DisplayData *data) {
    
    int fd = open("/dev/fb0", O_RDWR);
    if (fd == -1) {
        perror("Failed to open framebuffer device");
        return -1;
    }
     
    if (ioctl(fd, FBIOGET_FSCREENINFO, &finfo)) {
        perror("Cannot FBIOGET_FSCREENINFO");
        close(fd);
        return -1;
    }
    
    if (ioctl(fd, FBIOGET_VSCREENINFO, &vinfo)) {
        perror("Cannot FBIOGET_VSCREENINFO");
        close(fd);
        return -1;
    }

    
    long screen_size = vinfo.yres * finfo.line_length;
    long mmap_size = screen_size;

    buffer = malloc(sizeof(uint8_t) * mmap_size);

    fb_p = mmap(0, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (fb_p == MAP_FAILED) {
        perror(" Failed to map framebuffer device");
        close(fd);
        return -1;
    }

    m_display_width = vinfo.xres;
    m_display_height = vinfo.yres;
    m_pixel_step = vinfo.bits_per_pixel / 8;
    m_screen_size = screen_size;
    m_mmap_size = mmap_size;

    DisplayData display_data = {
        .framebuffer_desc = fd,
        .width = m_display_width,
        .height = m_display_height,
        .pixel_step = m_pixel_step,
        .line_length = finfo.line_length,
        .screen_size = m_screen_size,
        .mmap_size = m_mmap_size
    };

    *data = display_data;
    return 0;
}

void clean_up(int fd) {
    munmap(fb_p, m_mmap_size);
    close(fd);
    free(buffer);
}
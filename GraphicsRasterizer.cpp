
#include <windows.h>
#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include <cstring>
#include <thread>
#include <type_traits>

using namespace std::chrono_literals;


constexpr int CANVAS_WIDTH = 600;
constexpr int CANVAS_HEIGHT = 600;
constexpr int MAX_LINE_LENGTH = max(CANVAS_WIDTH, CANVAS_HEIGHT) + 50;
const int RECURSION_DEPTH = 3; // 0, 1, 2, 3, 5
std::vector<DWORD> canvasBuffer;
const float EPSILON = 0.001f;

// Scene setup.
constexpr int VIEWPORT_SIZE = 1;
constexpr int PROJECTION_PLANE_Z = 1;

enum class LightType {
    AMBIENT = 0,
    POINT = 1,
    DIRECTIONAL = 2,
};

struct PointOnCanvas {
    int x, y;
    float h;

    PointOnCanvas(int cx, int cy, float ch = 1.f) : x(cx), y(cy), h(ch) {}
    ~PointOnCanvas() {}
};

struct Vector3 {
    float x, y, z;

    template<typename T, typename U, typename V,
             std::enable_if_t<std::is_convertible<T, float>::value &&
                              std::is_convertible<U, float>::value &&
                              std::is_convertible<V, float>::value, bool> = true>
    Vector3(T cx, U cy, V cz):
        x(static_cast<float>(cx)),
        y(static_cast<float>(cy)),
        z(static_cast<float>(cz)) {}
};

struct Matrix3 {
    std::array<float, 9> matrix_buf;
    Vector3 line0 = { matrix_buf[0], matrix_buf[1], matrix_buf[2] };
    Vector3 line1 = { matrix_buf[3], matrix_buf[4], matrix_buf[5] };
    Vector3 line2 = { matrix_buf[6], matrix_buf[7], matrix_buf[8] };

    Vector3 col0 = { matrix_buf[0], matrix_buf[3], matrix_buf[6] };
    Vector3 col1 = { matrix_buf[1], matrix_buf[4], matrix_buf[7] };
    Vector3 col2 = { matrix_buf[2], matrix_buf[5], matrix_buf[8] };
};

struct Color {
    unsigned b, g, r;
    Color(unsigned cb, unsigned cg, unsigned cr) : b(cb), g(cg), r(cr) {}
    ~Color() {}
};

struct Sphere {
    Vector3 center;
    float radius;
    Color color;
    int specular;
    float reflective; // [0.f, 1.f]
};

struct Light {
    LightType ltype;
    float intensity;
    Vector3 position;
};

// Scene setup
const Vector3 CAMERA_POSITION = { 3, 0, 1 };

const Matrix3 CAMERA_ROTATION = {
    0.7071f, 0.f, -0.7071f,
    0.f, 1.f, 0.f,
    0.7071f, 0, 0.7071f
};

const std::vector<Sphere> SPHERES = {
    { {0, -1, 3}, 1, {0, 0, 255}, 500, 0.2f }, // red sphere
    { {2, 0, 4}, 1, {255, 0, 0}, 500, 0.3f }, // blue sphere
    { {-2, 0, 4 }, 1, { 0, 255, 0}, 10, 0.4f }, // green sphere
    { {0, -5001, 0}, 5000, {0, 255, 255}, 1000, 0.5f }, // yellow sphere
    { {0, 2, 2}, 2, {0, 255, 255}, 1000, 0.5f } // yellow sphere
};

// Lights setup
const std::vector<Light> LIGHTS = {
    { LightType::AMBIENT, 0.2f, {INFINITY, INFINITY, INFINITY} },
    { LightType::POINT, 0.6f, {2, 1, 0} },
    { LightType::DIRECTIONAL, 0.2f, {1, 4, 4} }
};

const Color BACKGROUND_COLOR = { 255, 255, 255 };

void PutPixel(int x, int y, const Color& color) {
    int x_r = CANVAS_WIDTH / 2 + x;
    int y_r = CANVAS_HEIGHT / 2 - y;

    if (x_r >= 0 && x_r < CANVAS_WIDTH && y_r >= 0 && y_r < CANVAS_HEIGHT) {
        int offset = x_r + CANVAS_WIDTH * y_r;
        canvasBuffer[offset] = RGB(color.b, color.g, color.r);
    }
}

void UpdateCanvas(HWND hwnd, HDC hdc, int CANVAS_WIDTH, int CANVAS_HEIGHT) {
    BITMAPINFO bmi;
    ZeroMemory(&bmi, sizeof(BITMAPINFO));

    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = CANVAS_WIDTH;
    bmi.bmiHeader.biHeight = -CANVAS_HEIGHT;  // Negative height to ensure top-down drawing
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    // Additional setup if using 32 bits per pixel
    bmi.bmiHeader.biClrUsed = 0;
    bmi.bmiHeader.biClrImportant = 0;

    SetDIBitsToDevice(hdc, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, 0, 0, 0,
        CANVAS_HEIGHT, canvasBuffer.data(), &bmi, DIB_RGB_COLORS);
}


// =============================================================================
//                         Vector3 operating routines
// =============================================================================

float DotProduct(const Vector3& v1, const Vector3& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

Vector3 Subtract(const Vector3& v1, const Vector3& v2) {
    return { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
}

// Length of a 3D vector.
float Length(const Vector3& v) {
    return std::sqrt(DotProduct(v, v));
}

// Computes k * vec.
Vector3 Multiply(float k, const Vector3& v) {
    return { k * v.x, k * v.y, k * v.z };
}

// Computes v1 + v2.
Vector3 Add(const Vector3& v1, const Vector3& v2) {
    return { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
}

// =============================================================================
//                            Color operating routines                
// =============================================================================

// Computes i * color.
Color Multiply(float i, const Color& c) {
    return { (unsigned)(i * c.b), (unsigned)(i * c.g), (unsigned)(i * c.r) };
}

// Computes color1 + color2.
Color Add(const Color& c1, const Color& c2) {
    return { c1.b + c2.b, c1.g + c2.g, c1.r + c2.r };
}

// Clamps a color to the canonical color range.
Color Clamp(const Color& c) {
    return {
        min(255, max(0, c.b)),
        min(255, max(0, c.g)),
        min(255, max(0, c.r))
    };
}

// =============================================================================
//                            Matrix operating routines                
// =============================================================================

// Multiplies a matrix and a vector.
Vector3 MultiplyMV(const Matrix3& mat, const Vector3& vec) {
    Vector3 result = { 0.f, 0.f, 0.f };

    result.x = DotProduct(mat.line0, vec);
    result.y = DotProduct(mat.line1, vec);
    result.z = DotProduct(mat.line2, vec);

    return result;
}


Vector3 ReflectRayDirection(const Vector3& ray, const Vector3& normal) {
    return Subtract(Multiply(2 * DotProduct(ray, normal), normal), ray);
}

struct InterpolateCache {
    float cache_[MAX_LINE_LENGTH] = {};
    size_t count_ = 0;

    void UpdateCache(float value) {
        if (count_ >= MAX_LINE_LENGTH)
            return;

        cache_[count_] = value;
        count_++;
    }

    float Get(size_t index) {
        return cache_[index];
    }

    void Reset() {
        count_ = 0;
    }

    void PopBack() {
        count_--;
    }

    size_t Size() {
        return count_;
    }

    void AddToTail(const InterpolateCache& tailCache) {
        if (count_ + tailCache.count_ >= MAX_LINE_LENGTH)
            return;

        memcpy(&cache_[count_], tailCache.cache_, tailCache.count_ * sizeof(float));
        count_ += tailCache.count_;
    }
};

template <typename T, typename U>
void Interpolate(T i0, U d0, T i1, U d1, InterpolateCache& ds) {
    if (i0 == i1) {
        ds.UpdateCache(static_cast<float>(d0));
    }

    float a = static_cast<float>(d1 - d0) / (i1 - i0);
    float d = static_cast<float>(d0);

    //ds.Reset();

    for (int i = i0; i <= i1; i++) {
        ds.UpdateCache(d);
        d += a;
    }
}

void DrawLine(const PointOnCanvas& p0, const PointOnCanvas& p1, const Color& color,
    InterpolateCache* cache) {
    int dx = p1.x - p0.x;
    int dy = p1.y - p0.y;
    //std::vector<float> ds;
    InterpolateCache& ds = *cache;

    const PointOnCanvas* pp0 = &p0;
    const PointOnCanvas* pp1 = &p1;

    if (std::abs(dx) > std::abs(dy)) {
        // The line is horizontal-ish. Make sure it's left to right.
        if (dx < 0) {
            auto swap = pp0;
            pp0 = pp1;
            pp1 = swap;
        }

        // Compute the Y values and draw.
        Interpolate((*pp0).x, (*pp0).y, (*pp1).x, (*pp1).y, ds);

        for (int x = (*pp0).x; x <= (*pp1).x; x++) {
            PutPixel(x, static_cast<int>(ds.Get(x - (*pp0).x)), color);
        }
    }
    else {
        // The line is verical-ish. Make sure it's bottom to top.
        if (dy < 0) {
            auto swap = pp0;
            pp0 = pp1;
            pp1 = swap;
        }

        // Compute the X values and draw.
        Interpolate((*pp0).y, (*pp0).x, (*pp1).y, (*pp1).x, ds);

        for (int y = (*pp0).y; y <= (*pp1).y; y++) {
            PutPixel(static_cast<int>(ds.Get(y - (*pp0).y)), y, color);
        }
    }

    ds.Reset();
}

void DrawWireframeTriangle(const PointOnCanvas& p0, const PointOnCanvas& p1,
    const PointOnCanvas& p2, const Color& color,
    InterpolateCache* cache) {
    DrawLine(p0, p1, color, cache);
    DrawLine(p1, p2, color, cache);
    DrawLine(p0, p2, color, cache);
}


void DrawTriangle(const PointOnCanvas& p0, const PointOnCanvas& p1,
    const PointOnCanvas& p2, const Color& color,
    InterpolateCache* cache) {
    const PointOnCanvas* pp0 = &p0;
    const PointOnCanvas* pp1 = &p1;
    const PointOnCanvas* pp2 = &p2;

    // Sort the points from bottom to top.
    if (p1.y < p0.y) {
        auto swap = pp0;
        pp0 = pp1;
        pp1 = swap;
    }
    if (p2.y < p0.y) {
        auto swap = pp0;
        pp0 = pp2;
        pp2 = swap;
    }
    if (p2.y < p1.y) {
        auto swap = pp1;
        pp1 = pp2;
        pp2 = swap;
    }

    InterpolateCache& x012 = cache[0];
    InterpolateCache& x02 = cache[1];
    InterpolateCache& h012 = cache[2];
    InterpolateCache& h02 = cache[3];

    // Compute X coordinates of the edges.
    Interpolate((*pp0).y, (*pp0).x, (*pp1).y, (*pp1).x, x012);
    Interpolate((*pp0).y, (*pp0).h, (*pp1).y, (*pp1).h, h012);
    x012.PopBack();
    h012.PopBack();
    Interpolate((*pp1).y, (*pp1).x, (*pp2).y, (*pp2).x, x012);
    Interpolate((*pp1).y, (*pp1).h, (*pp2).y, (*pp2).h, h012);

    Interpolate((*pp0).y, (*pp0).x, (*pp2).y, (*pp2).x, x02);
    Interpolate((*pp0).y, (*pp0).h, (*pp2).y, (*pp2).h, h02);

    // Determine which is left and which is right.
    InterpolateCache* x_left, * x_right;
    InterpolateCache* h_left, * h_right;
    int m = (x02.Size() / 2);
    if (x02.Get(m) < x012.Get(m)) {
        x_left = &x02;
        x_right = &x012;

        h_left = &h02;
        h_right = &h012;
    }
    else {
        x_left = &x012;
        x_right = &x02;

        h_left = &h012;
        h_right = &h02;
    }

    // Draw horizontal segments.
    for (int y = (*pp0).y; y <= (*pp2).y; y++) {
        int xl = static_cast<int>((*x_left).Get(y - (*pp0).y));
        int xr = static_cast<int>((*x_right).Get(y - (*pp0).y));

        InterpolateCache& h_segment = cache[4];

        Interpolate(xl, (*h_left).Get(y - (*pp0).y), xr, (*h_right).Get(y - (*pp0).y), h_segment);

        for (int x = xl; x <= xr; x++)
            PutPixel(x, y, Multiply(h_segment.Get(x - xl), color));

        h_segment.Reset();
    }

    x012.Reset();
    x02.Reset();
    h012.Reset();
    h02.Reset();
}


// Converts 2D viewport coordinates to 2D canvas coordinates.
PointOnCanvas ViewportToCanvas(float x, float y) {
    return PointOnCanvas(static_cast<int>(x * CANVAS_WIDTH / VIEWPORT_SIZE),
                         static_cast<int>(y * CANVAS_HEIGHT / VIEWPORT_SIZE));
}

PointOnCanvas ProjectVector3(const Vector3& v) {
    return ViewportToCanvas(v.x * PROJECTION_PLANE_Z / v.z,
                            v.y * PROJECTION_PLANE_Z / v.z);
}


//void RenderSection(int start_y, int end_y, const std::vector<Sphere>& spheres,
//    const std::vector<Light>& lights, const Matrix3& camera_rotation,
//    const Vector3& camera_position) {
//
//    for (int y = start_y; y < end_y; ++y) {
//        for (int x = -CANVAS_WIDTH / 2; x < CANVAS_WIDTH / 2; ++x) {
//            Vector3 direction = CanvasToViewport(x, y);
//            direction = MultiplyMV(camera_rotation, direction);
//
//            Color color = TraceRay(camera_position, direction, 1, INFINITY,
//                spheres, lights, RECURSION_DEPTH, nullptr/*, &cache*/);
//
//            PutPixel(x, y, Clamp(color));
//        }
//    }
//}

LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    HDC hdc;
    PAINTSTRUCT ps;

    switch (message) {
    case WM_PAINT:
        hdc = BeginPaint(hwnd, &ps);
        UpdateCanvas(hwnd, hdc, CANVAS_WIDTH, CANVAS_HEIGHT);
        EndPaint(hwnd, &ps);
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    default:
        return DefWindowProc(hwnd, message, wParam, lParam);
    }

    return 0;
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR lpCmdLine, _In_ int nCmdShow) {
    // Initialize canvas buffer
    canvasBuffer.resize(CANVAS_WIDTH * CANVAS_HEIGHT);

    // Determine the number of threads to use
    unsigned int num_threads = 16;// std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);
    int section_width = CANVAS_HEIGHT / num_threads;

    int nSpeed = 2;
    int nSpeedCount = 0;
    bool bChangePosition = false;
    Vector3 camera_pos = { 0.f, 0.f, 0.f };

    // initialize the canvasBuffer with white color as the background
    for (int x = -CANVAS_WIDTH / 2; x < CANVAS_WIDTH / 2; x++)
        for (int y = -CANVAS_HEIGHT / 2; y < CANVAS_HEIGHT / 2; y++)
            PutPixel(x, y, BACKGROUND_COLOR);

    auto start = std::chrono::high_resolution_clock::now(); // Start the timer

    auto p0 = PointOnCanvas(-200, -250, 0.3f);
    auto p1 = PointOnCanvas(200, 50, 0.1f);
    auto p2 = PointOnCanvas(20, 250, 1.f);

    std::vector<InterpolateCache> caches(5);
    //DrawTriangle(p0, p1, p2, Color(0, 255, 0), caches.data());
    //DrawWireframeTriangle(p0, p1, p2, Color(0, 0, 255), caches.data());

    auto vA = Vector3(-2.f, -0.5f, 5.f);
    auto vB = Vector3(-2.f, 0.5f, 5.f);
    auto vC = Vector3(-1.f, 0.5f, 5.f);
    auto vD = Vector3(-1.f, -0.5f, 5.f);

    auto vAb = Vector3(-2.f, -0.5f, 6.f);
    auto vBb = Vector3(-2.f, 0.5f, 6.f);
    auto vCb = Vector3(-1.f, 0.5f, 6.f);
    auto vDb = Vector3(-1.f, -0.5f, 6.f);

    auto BLUE = Color(255, 0, 0);
    auto GREEN = Color(0, 255, 0);
    auto RED = Color(0, 0, 255);

    DrawLine(ProjectVector3(vA), ProjectVector3(vB), BLUE, caches.data());
    DrawLine(ProjectVector3(vB), ProjectVector3(vC), BLUE, caches.data());
    DrawLine(ProjectVector3(vC), ProjectVector3(vD), BLUE, caches.data());
    DrawLine(ProjectVector3(vD), ProjectVector3(vA), BLUE, caches.data());

    DrawLine(ProjectVector3(vAb), ProjectVector3(vBb), RED, caches.data());
    DrawLine(ProjectVector3(vBb), ProjectVector3(vCb), RED, caches.data());
    DrawLine(ProjectVector3(vCb), ProjectVector3(vDb), RED, caches.data());
    DrawLine(ProjectVector3(vDb), ProjectVector3(vAb), RED, caches.data());

    DrawLine(ProjectVector3(vA), ProjectVector3(vAb), GREEN, caches.data());
    DrawLine(ProjectVector3(vB), ProjectVector3(vBb), GREEN, caches.data());
    DrawLine(ProjectVector3(vC), ProjectVector3(vCb), GREEN, caches.data());
    DrawLine(ProjectVector3(vD), ProjectVector3(vDb), GREEN, caches.data());

    /*// Create threads to render sections of the canvas
    for (unsigned int i = 0; i < num_threads; ++i) {
        int start_y = i * section_width - CANVAS_HEIGHT / 2;
        int end_y = (i + 1) * section_width - CANVAS_HEIGHT / 2;
        if (i == num_threads - 1) {
            end_y = CANVAS_HEIGHT / 2;
        }

        threads[i] = std::thread(RenderSection, start_y, end_y,
            std::cref(SPHERES), std::cref(LIGHTS), std::cref(CAMERA_ROTATION),
            std::cref(CAMERA_POSITION));
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }*/

    auto end = std::chrono::high_resolution_clock::now(); // Stop the timer
    auto duration = // Calculate the duration in milliseconds
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::string time_str = // Convert the duration to a string
        "Time: " + std::to_string(duration.count()) + " milliseconds";

    // Register the window class
    WNDCLASS wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = L"RaytracerDemo";

    RegisterClass(&wc);

    // Create the window
    HWND hwnd = CreateWindowEx(0, L"RaytracerDemo", L"Raytracer Demo",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, CANVAS_WIDTH,
        CANVAS_HEIGHT + 40, nullptr, nullptr, wc.hInstance, nullptr);

    if (hwnd == nullptr) {
        MessageBox(nullptr, L"Window creation failed", L"Error", MB_ICONERROR);
        return 1;
    }

    ShowWindow(hwnd, SW_SHOWNORMAL);

    // Set the time_str as the window text
    SetWindowTextA(hwnd, time_str.c_str());

    // Main loop
    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    /*bool running = true;

    while (running) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                running = false;
                break;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        std::this_thread::sleep_for(10ms);

        // Create threads to render sections of the canvas
        for (unsigned int i = 0; i < num_threads; ++i) {
            int start_y = i * section_width - CANVAS_HEIGHT / 2;
            int end_y = (i + 1) * section_width - CANVAS_HEIGHT / 2;
            if (i == num_threads - 1) {
                end_y = CANVAS_HEIGHT / 2;
            }

            threads[i] = std::thread(RenderSection, start_y, end_y,
                std::cref(SPHERES), std::cref(LIGHTS), std::cref(CAMERA_ROTATION),
                std::cref(camera_pos));
        }

        // Join threads
        for (auto& thread : threads) {
            thread.join();
        }

        nSpeedCount++;
        bChangePosition = (nSpeedCount == nSpeed);

        if (bChangePosition) {
            nSpeedCount = 0;
            camera_pos.x += 0.005f;
            camera_pos.y += 0.001f;
            camera_pos.z -= 0.001f;
        }

        // Update canvas
        HDC hdc = GetDC(hwnd);
        UpdateCanvas(hwnd, hdc, CANVAS_WIDTH, CANVAS_HEIGHT);
        ReleaseDC(hwnd, hdc);
    } */

    return 0;
}

#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <string>
#include <thread>
#include <windows.h>

using namespace std::chrono_literals;


const int CANVAS_WIDTH = 600;
const int CANVAS_HEIGHT = 600;
const int RECURSION_DEPTH = 3; // 0, 1, 2, 3, 5
std::vector<DWORD> canvasBuffer;
const float EPSILON = 0.001f;

enum class LightType {
    AMBIENT = 0,
    POINT = 1,
    DIRECTIONAL = 2,
};

struct PointOnCanvas {
    int x, y;

    PointOnCanvas(int cx, int cy) : x(cx), y(cy) {}
    ~PointOnCanvas() {}
};

struct Vector3 {
    float x, y, z;
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


void Interpolate(int i0, int d0, int i1, int d1, std::vector<float>& ds) {
    if (i0 == i1) {
        ds.push_back(static_cast<float>(d0));
    }

    float a = static_cast<float>(d1 - d0) / (i1 - i0);
    float d = static_cast<float>(d0);

    for (int i = i0; i <= i1; i++) {
        ds.push_back(d);
        d += a;
    }
}

void DrawLine(const PointOnCanvas& p0, const PointOnCanvas& p1, const Color& color) {
    int dx = p1.x - p0.x;
    int dy = p1.y - p0.y;
    std::vector<float> ds;

    if (std::abs(dx) > std::abs(dy)) {
        // The line is horizontal-ish. Make sure it's left to right.
        if (dx < 0) {
            auto swap = p0;
            const_cast<PointOnCanvas&>(p0) = p1;
            const_cast<PointOnCanvas&>(p1) = swap;
        }

        // Compute the Y values and draw.
        Interpolate(p0.x, p0.y, p1.x, p1.y, ds);

        for (int x = p0.x; x <= p1.x; x++) {
            PutPixel(x, static_cast<int>(ds[(x - p0.x) | 0]), color);
        }
    }
    else {
        // The line is verical-ish. Make sure it's bottom to top.
        if (dy < 0) {
            auto swap = p0;
            const_cast<PointOnCanvas&>(p0) = p1;
            const_cast<PointOnCanvas&>(p1) = swap;
        }

        // Compute the X values and draw.
        Interpolate(p0.y, p0.x, p1.y, p1.x, ds);

        for (int y = p0.y; y <= p1.y; y++) {
            PutPixel(static_cast<int>(ds[(y - p0.y) | 0]), y, color);
        }
    }
}

void DrawWireframeTriangle(const PointOnCanvas& p0, const PointOnCanvas& p1, const PointOnCanvas& p2, const Color& color) {
    DrawLine(p0, p1, color);
    DrawLine(p1, p2, color);
    DrawLine(p0, p2, color);
}


void DrawFilledTriangle(const PointOnCanvas& p0, const PointOnCanvas& p1, const PointOnCanvas& p2, const Color& color) {
    // Sort the points from bottom to top.
    if (p1.y < p0.y) {
        auto swap = p0;
        const_cast<PointOnCanvas&>(p0) = p1;
        const_cast<PointOnCanvas&>(p1) = swap;
    }
    if (p2.y < p0.y) {
        auto swap = p0;
        const_cast<PointOnCanvas&>(p0) = p2;
        const_cast<PointOnCanvas&>(p2) = swap;
    }
    if (p2.y < p1.y) {
        auto swap = p1;
        const_cast<PointOnCanvas&>(p1) = p2;
        const_cast<PointOnCanvas&>(p2) = swap;
    }

    std::vector<float> x01;
    std::vector<float> x12;
    std::vector<float> x02;

    // Compute X coordinates of the edges.
    Interpolate(p0.y, p0.x, p1.y, p1.x, x01);
    Interpolate(p1.y, p1.x, p2.y, p2.x, x12);
    Interpolate(p0.y, p0.x, p2.y, p2.x, x01);

    // Merge the two short sides.
    x01.pop_back();
    x01.insert(x01.end(), x12.begin(), x12.end());

    // Determine which is left and which is right.
    std::vector<float> *x_left, *x_right;
    int m = (x02.size() / 2);
    if (x02[m] < x01[m]) {
        x_left = &x02;
        x_right = &x01;
    }
    else {
        x_left = &x01;
        x_right = &x02;
    }

    // Draw horizontal segments.
    for (int y = p0.y; y <= p2.y; y++) {
        for (int x = static_cast<int>((*x_left)[y - p0.y]); x <= static_cast<int>((*x_right)[y - p0.y]); x++) {
            PutPixel(x, y, color);
        }
    }
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

    /*DrawLine(PointOnCanvas(-200, -100), PointOnCanvas(240, 120), Color(0, 0, 0));
    DrawLine(PointOnCanvas(-50, -200), PointOnCanvas(60, 240), Color(0, 0, 0));
    DrawLine(PointOnCanvas(-50, -100), PointOnCanvas(-50, 100), Color(0, 0, 0));
    DrawLine(PointOnCanvas(-50, -50), PointOnCanvas(60, 60), Color(0, 0, 0));
    DrawLine(PointOnCanvas(-150, 100), PointOnCanvas(150, 100), Color(0, 0, 0));*/

    auto p0 = PointOnCanvas(-200, -250);
    auto p1 = PointOnCanvas(200, 50);
    auto p2 = PointOnCanvas(20, 250);

    DrawFilledTriangle(p0, p1, p2, Color(0, 255, 0));
    DrawWireframeTriangle(p0, p1, p2, Color(0, 0, 0));

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

#include <time.h>

#include "imgui.h"
#include "imgui_impl_xlib.h"

timespec _get_wall_clock() {
    timespec result;
    clock_gettime(CLOCK_MONOTONIC, &result);
    return result;
}

float _get_seconds_elapsed(timespec start, timespec end) {
    float delta_seconds = (float) (end.tv_sec - start.tv_sec);
    float delta_nanoseconds = (float) (end.tv_nsec - start.tv_nsec);
    return delta_seconds + delta_nanoseconds * 1.0e-9f;
}

// Xlib Data
struct ImGui_ImplXlib_Data
{
    Display*        display;
    Window          window;
    timespec        Time;
    int             MouseButtonsDown;
    int             PendingMouseLeaveFrame;
    char*           ClipboardTextData;
    bool            MouseCanUseGlobalState;

    ImGui_ImplXlib_Data()   { memset((void*)this, 0, sizeof(*this)); }
};

static ImGui_ImplXlib_Data* ImGui_ImplXlib_GetBackendData()
{
    return ImGui::GetCurrentContext() ? (ImGui_ImplXlib_Data*)ImGui::GetIO().BackendPlatformUserData : NULL;
}

// Functions
static const char* ImGui_ImplXlib_GetClipboardText(void*)
{
    // TODO
    return "";
}

static void ImGui_ImplXlib_SetClipboardText(void*, const char* text)
{
    // TODO
}

static ImGuiKey ImGui_ImplXlib_KeycodeToImGuiKey(int keycode)
{
    switch (keycode)
    {
        case XK_Tab: return ImGuiKey_Tab;
        case XK_Left: return ImGuiKey_LeftArrow;
        case XK_Right: return ImGuiKey_RightArrow;
        case XK_Up: return ImGuiKey_UpArrow;
        case XK_Down: return ImGuiKey_DownArrow;
        case XK_Page_Up: return ImGuiKey_PageUp;
        case XK_Page_Down: return ImGuiKey_PageDown;
        case XK_Home: return ImGuiKey_Home;
        case XK_End: return ImGuiKey_End;
        case XK_Insert: return ImGuiKey_Insert;
        case XK_Delete: return ImGuiKey_Delete;
        case XK_BackSpace: return ImGuiKey_Backspace;
        case XK_space: return ImGuiKey_Space;
        case XK_Return: return ImGuiKey_Enter;
        case XK_Escape: return ImGuiKey_Escape;
        case XK_apostrophe: return ImGuiKey_Apostrophe;
        case XK_comma: return ImGuiKey_Comma;
        case XK_minus: return ImGuiKey_Minus;
        case XK_period: return ImGuiKey_Period;
        case XK_slash: return ImGuiKey_Slash;
        case XK_semicolon: return ImGuiKey_Semicolon;
        case XK_equal: return ImGuiKey_Equal;
        case XK_bracketleft: return ImGuiKey_LeftBracket;
        case XK_backslash: return ImGuiKey_Backslash;
        case XK_bracketright: return ImGuiKey_RightBracket;
        case XK_quotedbl: return ImGuiKey_GraveAccent;
        case XK_Caps_Lock: return ImGuiKey_CapsLock;
        case XK_Scroll_Lock: return ImGuiKey_ScrollLock;
        case XK_Num_Lock: return ImGuiKey_NumLock;
        case XK_Print: return ImGuiKey_PrintScreen;
        case XK_Pause: return ImGuiKey_Pause;
        case XK_KP_0: return ImGuiKey_Keypad0;
        case XK_KP_1: return ImGuiKey_Keypad1;
        case XK_KP_2: return ImGuiKey_Keypad2;
        case XK_KP_3: return ImGuiKey_Keypad3;
        case XK_KP_4: return ImGuiKey_Keypad4;
        case XK_KP_5: return ImGuiKey_Keypad5;
        case XK_KP_6: return ImGuiKey_Keypad6;
        case XK_KP_7: return ImGuiKey_Keypad7;
        case XK_KP_8: return ImGuiKey_Keypad8;
        case XK_KP_9: return ImGuiKey_Keypad9;
        case XK_KP_Decimal: return ImGuiKey_KeypadDecimal;
        case XK_KP_Divide: return ImGuiKey_KeypadDivide;
        case XK_KP_Multiply: return ImGuiKey_KeypadMultiply;
        case XK_KP_Subtract: return ImGuiKey_KeypadSubtract;
        case XK_KP_Add: return ImGuiKey_KeypadAdd;
        case XK_KP_Enter: return ImGuiKey_KeypadEnter;
        case XK_KP_Equal: return ImGuiKey_KeypadEqual;
        case XK_Control_L: return ImGuiKey_LeftCtrl;
        case XK_Shift_L: return ImGuiKey_LeftShift;
        case XK_Alt_L: return ImGuiKey_LeftAlt;
        case XK_Super_L: return ImGuiKey_LeftSuper;
        case XK_Control_R: return ImGuiKey_RightCtrl;
        case XK_Shift_R: return ImGuiKey_RightShift;
        case XK_Alt_R: return ImGuiKey_RightAlt;
        case XK_Super_R: return ImGuiKey_RightSuper;
        case XK_Menu: return ImGuiKey_Menu;
        case XK_0: return ImGuiKey_0;
        case XK_1: return ImGuiKey_1;
        case XK_2: return ImGuiKey_2;
        case XK_3: return ImGuiKey_3;
        case XK_4: return ImGuiKey_4;
        case XK_5: return ImGuiKey_5;
        case XK_6: return ImGuiKey_6;
        case XK_7: return ImGuiKey_7;
        case XK_8: return ImGuiKey_8;
        case XK_9: return ImGuiKey_9;
        case XK_a: return ImGuiKey_A;
        case XK_b: return ImGuiKey_B;
        case XK_c: return ImGuiKey_C;
        case XK_d: return ImGuiKey_D;
        case XK_e: return ImGuiKey_E;
        case XK_f: return ImGuiKey_F;
        case XK_g: return ImGuiKey_G;
        case XK_h: return ImGuiKey_H;
        case XK_i: return ImGuiKey_I;
        case XK_j: return ImGuiKey_J;
        case XK_k: return ImGuiKey_K;
        case XK_l: return ImGuiKey_L;
        case XK_m: return ImGuiKey_M;
        case XK_n: return ImGuiKey_N;
        case XK_o: return ImGuiKey_O;
        case XK_p: return ImGuiKey_P;
        case XK_q: return ImGuiKey_Q;
        case XK_r: return ImGuiKey_R;
        case XK_s: return ImGuiKey_S;
        case XK_t: return ImGuiKey_T;
        case XK_u: return ImGuiKey_U;
        case XK_v: return ImGuiKey_V;
        case XK_w: return ImGuiKey_W;
        case XK_x: return ImGuiKey_X;
        case XK_y: return ImGuiKey_Y;
        case XK_z: return ImGuiKey_Z;
        case XK_F1: return ImGuiKey_F1;
        case XK_F2: return ImGuiKey_F2;
        case XK_F3: return ImGuiKey_F3;
        case XK_F4: return ImGuiKey_F4;
        case XK_F5: return ImGuiKey_F5;
        case XK_F6: return ImGuiKey_F6;
        case XK_F7: return ImGuiKey_F7;
        case XK_F8: return ImGuiKey_F8;
        case XK_F9: return ImGuiKey_F9;
        case XK_F10: return ImGuiKey_F10;
        case XK_F11: return ImGuiKey_F11;
        case XK_F12: return ImGuiKey_F12;
    }
    return ImGuiKey_None;
}

bool ImGui_ImplXlib_ProcessEvent(const XEvent* event)
{
    ImGuiIO& io = ImGui::GetIO();
    ImGui_ImplXlib_Data* bd = ImGui_ImplXlib_GetBackendData();

    switch (event->type)
    {
        case KeyRelease:
        case KeyPress:
        {
            XKeyEvent *key_event = (XKeyEvent *)event;
            KeySym key_sym = XLookupKeysym(key_event, 0);
            ImGuiKey key = ImGui_ImplXlib_KeycodeToImGuiKey(key_sym);

            // TODO: Key modifiers (ctrl, shift, alt, super)

            io.AddKeyEvent(key, (event->type == KeyPress));
            if (event->type == KeyPress && 0 < key_sym && key_sym < 0xff) { io.AddInputCharacter(key_sym); }
            return true;
        }
        case MotionNotify:
        {
            XMotionEvent *motion_event = (XMotionEvent *)event;
            io.AddMousePosEvent((float)motion_event->x, (float)motion_event->y);
            return true;
        }
        case ButtonPress:
        case ButtonRelease:
        {
            int mouse_button = -1;
            XButtonPressedEvent *button_event = (XButtonPressedEvent *)event;
            if (button_event->button == Button1) { mouse_button = 0; }
            if (button_event->button == Button2) { mouse_button = 1; }
            if (button_event->button == Button3) { mouse_button = 2; }
            if (button_event->button == Button4) { mouse_button = 3; }
            if (button_event->button == Button5) { mouse_button = 4; }
            if (mouse_button == -1)
                break;
            io.AddMouseButtonEvent(mouse_button, (event->type == ButtonPress));
            bd->MouseButtonsDown = (event->type == ButtonPress) ? (bd->MouseButtonsDown | (1 << mouse_button)) : (bd->MouseButtonsDown & ~(1 << mouse_button));
            return true;
        }
    }
    return false;
}

static bool ImGui_ImplXlib_Init(Display *display, Window window)
{
    ImGuiIO& io = ImGui::GetIO();
    IM_ASSERT(io.BackendPlatformUserData == NULL && "Already initialized a platform backend!");

    // Setup backend capabilities flags
    ImGui_ImplXlib_Data* bd = IM_NEW(ImGui_ImplXlib_Data)();
    io.BackendPlatformUserData = (void*)bd;
    io.BackendPlatformName = "imgui_impl_xlib";

    bd->display = display;
    bd->window = window;
    bd->Time = _get_wall_clock();

    io.SetClipboardTextFn = ImGui_ImplXlib_SetClipboardText;
    io.GetClipboardTextFn = ImGui_ImplXlib_GetClipboardText;
    io.ClipboardUserData = NULL;
    return true;
}

bool ImGui_ImplXlib_InitForOpenGL(Display *display, Window window, void* sdl_gl_context)
{
    IM_UNUSED(sdl_gl_context); // Viewport branch will need this.
    return ImGui_ImplXlib_Init(display, window);
}

void ImGui_ImplXlib_Shutdown()
{
    ImGui_ImplXlib_Data* bd = ImGui_ImplXlib_GetBackendData();
    IM_ASSERT(bd != NULL && "No platform backend to shutdown, or already shutdown?");
    ImGuiIO& io = ImGui::GetIO();
    io.BackendPlatformName = NULL;
    io.BackendPlatformUserData = NULL;
    IM_DELETE(bd);
}

static void ImGui_ImplXlib_UpdateMouseData()
{
    // TODO
}

static void ImGui_ImplXlib_UpdateMouseCursor()
{
    // TODO
}

static void ImGui_ImplXlib_UpdateGamepads()
{
    // TODO
}

void ImGui_ImplXlib_NewFrame()
{
    ImGui_ImplXlib_Data* bd = ImGui_ImplXlib_GetBackendData();
    IM_ASSERT(bd != NULL && "Did you call ImGui_ImplXlib_Init()?");
    ImGuiIO& io = ImGui::GetIO();

    // Setup display size (every frame to accommodate for window resizing)
    int w, h;
    int display_w, display_h;
    XWindowAttributes window_attributes = {};
    XGetWindowAttributes(bd->display, bd->window, &window_attributes);
    display_w = w = window_attributes.width;
    display_h = h = window_attributes.height;

    io.DisplaySize = ImVec2((float)w, (float)h);
    if (w > 0 && h > 0)
        io.DisplayFramebufferScale = ImVec2((float)display_w / w, (float)display_h / h);

    timespec current_time = _get_wall_clock();
    io.DeltaTime = _get_seconds_elapsed(bd->Time, current_time);
    bd->Time = current_time;

    if (bd->PendingMouseLeaveFrame && bd->PendingMouseLeaveFrame >= ImGui::GetFrameCount() && bd->MouseButtonsDown == 0)
    {
        io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
        bd->PendingMouseLeaveFrame = 0;
    }

    ImGui_ImplXlib_UpdateMouseData();
    ImGui_ImplXlib_UpdateMouseCursor();

    // Update game controllers (if enabled and available)
    ImGui_ImplXlib_UpdateGamepads();
}

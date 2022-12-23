#pragma once
#include "imgui.h"      // IMGUI_IMPL_API

#include <X11/Xlib.h>

#define XK_MISCELLANY
#define XK_LATIN1
#include <X11/keysymdef.h>

IMGUI_IMPL_API bool     ImGui_ImplXlib_InitForOpenGL(Display *display, Window window, void* gl_context);
IMGUI_IMPL_API void     ImGui_ImplXlib_Shutdown();
IMGUI_IMPL_API void     ImGui_ImplXlib_NewFrame();
IMGUI_IMPL_API bool     ImGui_ImplXlib_ProcessEvent(const XEvent* event);

#ifndef IMGUI_DISABLE_OBSOLETE_FUNCTIONS
static inline void ImGui_ImplXlib_NewFrame(Window) { ImGui_ImplXlib_NewFrame(); } // 1.84: removed unnecessary parameter
#endif

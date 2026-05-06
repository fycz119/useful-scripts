-- 引入 wezterm API
local wezterm = require 'wezterm'

-- 使用 config_builder 提供更好的错误提示
local config = wezterm.config_builder()

-- ==========================================
-- 外观与主题设置
-- ==========================================

-- 设置颜色主题 (你可以在 wezterm 官网查找你喜欢的主题，例如 'Catppuccin Mocha', 'Tokyo Night' 等)
config.color_scheme = 'Dracula'

-- 设置字体 (需要你的电脑上已经安装了该字体，推荐使用带 Nerd Font 的编程字体)
config.font = wezterm.font('JetBrains Mono', { weight = 'Regular' })
config.font_size = 13.0

-- ==========================================
-- 窗口设置
-- ==========================================

-- 背景透明度 (0.0 完全透明, 1.0 完全不透明)
config.window_background_opacity = 0.90
-- macOS 下开启毛玻璃背景模糊效果
config.macos_window_background_blur = 20 

-- 窗口装饰类型 (隐藏原生的标题栏，让界面更沉浸)
config.window_decorations = "RESIZE"

-- 窗口内边距
config.window_padding = {
  left = '1cell',
  right = '1cell',
  top = '0.5cell',
  bottom = '0.5cell',
}

-- ==========================================
-- 标签页 (Tab Bar) 设置
-- ==========================================

-- 当只有一个标签页时隐藏标签栏
config.hide_tab_bar_if_only_one_tab = true
-- 使用复古样式的标签栏（关闭 fancy 样式通常在非原生标题栏下看起来更整洁）
config.use_fancy_tab_bar = false
config.tab_bar_at_bottom = true

-- ==========================================
-- 快捷键与行为设置
-- ==========================================

-- 将默认的前导键 (Leader Key) 设置为 Ctrl+A (类似 tmux)
config.leader = { key = 'a', mods = 'CTRL', timeout_milliseconds = 1000 }

-- 自定义快捷键
config.keys = {
  -- 按下 Leader + c 创建新标签页
  {
    key = 'c',
    mods = 'LEADER',
    action = wezterm.action.SpawnTab 'CurrentPaneDomain',
  },
  -- 按下 Leader + x 关闭当前面板
  {
    key = 'x',
    mods = 'LEADER',
    action = wezterm.action.CloseCurrentPane { confirm = true },
  },
  -- 水平分割面板: Leader + -
  {
    key = '-',
    mods = 'LEADER',
    action = wezterm.action.SplitVertical { domain = 'CurrentPaneDomain' },
  },
  -- 垂直分割面板: Leader + \
  {
    key = '\\',
    mods = 'LEADER',
    action = wezterm.action.SplitHorizontal { domain = 'CurrentPaneDomain' },
  },
}

-- 返回最终配置
return config

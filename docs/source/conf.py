# Configuration file for the Sphinx documentation builder.
import os
import sys

# -- Path setup --------------------------------------------------------------
# 指向你的项目根目录，以便 sphinx.ext.autodoc 能自动提取代码注释生成 API 文档
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
html_title = "DeepSpatial"
copyright = '2026, Yuhang Yang'
author = 'Yuhang Yang'

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",               # 解析 Markdown (.md) 和 Jupyter Notebook (.ipynb)
    "sphinx_copybutton",     # 代码块右上角的一键复制按钮
    "sphinx.ext.autodoc",    # 自动从 Python 代码中提取文档
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",   # 支持 NumPy 和 Google 风格的 Docstrings
    "sphinx.ext.viewcode",   # 在文档中添加 [source] 链接以查看源代码
    "sphinx.ext.intersphinx",# 允许链接到其他项目的文档（如 Scanpy, NumPy）
]

# 告诉 Sphinx 哪些文件需要被解析
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
}

templates_path = ['_templates']
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST-NB Configuration ---------------------------------------------------
# 关键设置：关闭编译时的代码执行。
# 这样 Sphinx 会直接读取你 .ipynb 文件中已有的输出结果和图片，而不会重新运行耗时的 3D 重建。
nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------

# 使用 Sphinx Book Theme (CellCharter 同款学术风主题)
html_theme = 'sphinx_book_theme'

# 浏览器标签页图标 (Favicon)
html_favicon = '_static/logo.png'

html_theme_options = {
    # 右上角的 GitHub 仓库按钮配置
    "repository_url": "https://github.com/yyh030806/DeepSpatial",
    "use_repository_button": True,
    "path_to_docs": "docs/source",
    
    # 顶部功能按钮
    "use_fullscreen_button": True,
    "use_download_button": True,
    
    # 左侧侧边栏配置
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "logo": {
        "text": "",         # Logo 旁显示的文字
        "image_light": "text_logo.png",     # 浅色模式 Logo
        "image_dark": "text_logo.png",      # 深色模式 Logo
    },
}

# 静态资源路径
html_static_path = ['_static']

# 引入自定义配色 CSS
html_css_files = [
    'custom.css',
]

# -- MyST Markdown 扩展配置 ---------------------------------------------------
myst_enable_extensions = [
    "colon_fence",    # 允许使用 ::: 这种警告框语法
    "amsmath",        # 支持复杂的 LaTeX 数学环境
    "dollarmath",     # 支持 $...$ 和 $$...$$ 数学公式
    "html_image",     # 允许在 Markdown 中直接使用 HTML <img> 标签
    "attr_list",      # 允许为 Markdown 元素添加 CSS 属性
]

# 设置公式自动编号
numfig = True
math_numfig = True
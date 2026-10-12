from simplechat.markdown import render_markdown


def test_basic_markdown() -> None:
    html = render_markdown("**bold** and ~~gone~~\nnext line\n\n- item")
    assert html == (
        "<p><strong>bold</strong> and <s>gone</s><br />\nnext line</p>\n"
        "<ul>\n<li>item</li>\n</ul>\n"
    )


def test_tables() -> None:
    html = render_markdown("| a | b |\n|---|---|\n| 1 | 2 |")
    assert "<th>a</th>" in html
    assert "<td>2</td>" in html


def test_raw_html_is_escaped() -> None:
    html = render_markdown('<script>alert(1)</script>\n\n<img src=x onerror="alert(1)">')
    assert "<script" not in html
    assert "<img" not in html
    assert "&lt;script&gt;" in html


def test_links_open_in_a_new_tab_and_unsafe_ones_are_dropped() -> None:
    html = render_markdown("[ok](https://example.com) [bad](javascript:alert(1))")
    assert '<a href="https://example.com" target="_blank" rel="noopener noreferrer">ok</a>' in html
    # The unsafe link stays as its Markdown text.
    assert "[bad](javascript:alert(1))" in html
    assert html.count("<a ") == 1


def test_images_are_not_loaded() -> None:
    html = render_markdown("![chart](https://example.com/x.png)")
    assert "<img" not in html


def test_code_blocks_get_a_header_with_copy_button() -> None:
    html = render_markdown('```py"><b>\nif a < b:\n```')
    assert '<span class="code-language">py&quot;&gt;&lt;b&gt;</span>' in html
    assert '<button class="copy-code" type="button">Copy</button>' in html
    assert "if a &lt; b:" in html

    indented = render_markdown("    x = 1")
    assert '<span class="code-language"></span>' in indented
    assert "<pre><code>x = 1\n</code></pre>" in indented


def test_unclosed_fence_while_streaming() -> None:
    html = render_markdown("Here:\n\n```py\nprint(1)")
    assert "<pre><code" in html
    assert "print(1)" in html

"""Render the model's Markdown replies to HTML."""

from collections.abc import Sequence

from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml, unescapeAll
from markdown_it.renderer import RendererHTML
from markdown_it.token import Token
from markdown_it.utils import EnvType, OptionsDict
from markupsafe import Markup

# html=False escapes raw HTML in a reply instead of passing it through, and
# markdown-it drops javascript: and similar links, so the output is safe to embed.
# breaks=True keeps single newlines as line breaks, as the plain-text view did.
_md = (
    MarkdownIt("commonmark", {"html": False, "breaks": True})
    .enable(["table", "strikethrough"])
    # Images would load from any URL the model writes, which leaks the chat to
    # that server and fails on a LAN without internet; `![alt](url)` stays text.
    .disable("image")
)


def _link_open(
    self: RendererHTML, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType
) -> str:
    # Opening a link in this tab would leave the chat.
    tokens[idx].attrSet("target", "_blank")
    tokens[idx].attrSet("rel", "noopener noreferrer")
    return self.renderToken(tokens, idx, options, env)


def _code_block(
    self: RendererHTML, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType
) -> str:
    """A fenced or indented code block, under a header with its language and a Copy button."""
    token = tokens[idx]
    if token.type == "fence":
        code = RendererHTML.fence(self, tokens, idx, options, env)
        info = unescapeAll(token.info).strip()
        language = info.split(maxsplit=1)[0] if info else ""
    else:
        code = RendererHTML.code_block(self, tokens, idx, options, env)
        language = ""
    # app.js copies the text of the block's <code> when the button is pressed.
    return (
        '<div class="code-block"><div class="code-head">'
        f'<span class="code-language">{escapeHtml(language)}</span>'
        '<button class="copy-code" type="button">Copy</button>'
        f"</div>{code}</div>\n"
    )


_md.add_render_rule("link_open", _link_open)
_md.add_render_rule("fence", _code_block)
_md.add_render_rule("code_block", _code_block)


def render_markdown(text: str) -> Markup:
    return Markup(_md.render(text))

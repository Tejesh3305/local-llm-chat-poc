/*
 * Markdown to HTML. Model output is untrusted: everything is escaped,
 * and links are shown as text because there is nowhere to go offline.
 */

const COPY_ICON =
  '<svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>';

const ESCAPES: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const escapeHtml = (text: string) => text.replace(/[&<>"']/g, c => ESCAPES[c]);

export function inlineMd(source: string): string {
  const spans: string[] = [];
  let text = source.replace(/`([^`\n]+)`/g, (_, code: string) => `\u0000${spans.push(code) - 1}\u0000`);
  text = escapeHtml(text)
    .replace(/\*\*(?=\S)(.+?)(?<=\S)\*\*/g, '<strong>$1</strong>')
    .replace(/__(?=\S)(.+?)(?<=\S)__/g, '<strong>$1</strong>')
    .replace(/(^|[^*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![*\w])/g, '$1<em>$2</em>')
    .replace(/(^|[^_\w])_(?!\s)([^_\n]+?)(?<!\s)_(?![_\w])/g, '$1<em>$2</em>')
    .replace(/~~(?=\S)(.+?)(?<=\S)~~/g, '<del>$1</del>')
    .replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, '<span class="link" title="$2">$1</span>');
  return text.replace(/\u0000(\d+)\u0000/g, (_, index: string) => `<code>${escapeHtml(spans[+index])}</code>`);
}

const RE_FENCE = /^(\s*)(`{3,}|~{3,})\s*([^\s`]*)[^`]*$/;
const RE_LIST = /^(\s*)([-*+•]|\d{1,3}[.)])\s+(.*)$/;
const RE_HEADING = /^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/;
const RE_RULE = /^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/;
const RE_QUOTE = /^\s{0,3}>\s?(.*)$/;
const RE_TABLE_RULE = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;

const indentOf = (line: string) => line.search(/\S|$/);
const isTableStart = (lines: string[], i: number) =>
  lines[i].includes('|') && i + 1 < lines.length && lines[i + 1].includes('|') && RE_TABLE_RULE.test(lines[i + 1]);
const startsBlock = (lines: string[], i: number) =>
  RE_FENCE.test(lines[i]) || RE_HEADING.test(lines[i]) || RE_RULE.test(lines[i]) || RE_QUOTE.test(lines[i]) ||
  RE_LIST.test(lines[i]) || isTableStart(lines, i);

/** Returns one HTML string per top-level block, each with a single root element. */
export function mdBlocks(source: string): string[] {
  return renderLines(source.replace(/\r\n?/g, '\n').replace(/\t/g, '    ').split('\n'));
}

function renderLines(lines: string[]): string[] {
  const out: string[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    let match: RegExpMatchArray | null;
    if (!line.trim()) { i++; continue; }

    if ((match = line.match(RE_FENCE))) {
      const indent = match[1].length;
      const closing = match[2][0] === '`' ? /^\s*`{3,}\s*$/ : /^\s*~{3,}\s*$/;
      const code: string[] = [];
      i++;
      while (i < lines.length && !closing.test(lines[i])) {
        code.push(lines[i].slice(Math.min(indent, indentOf(lines[i]))));
        i++;
      }
      i++;   // the closing fence, or the end of the text while a reply is still streaming
      out.push(
        `<div class="code"><div class="code-head"><span>${escapeHtml(match[3] || 'code')}</span>` +
        `<button type="button" class="ghost" data-copy="code">${COPY_ICON}<span>Copy</span></button></div>` +
        `<pre><code>${escapeHtml(code.join('\n'))}</code></pre></div>`,
      );
      continue;
    }

    if ((match = line.match(RE_HEADING))) {
      const level = match[1].length <= 2 ? 3 : match[1].length === 3 ? 4 : 5;
      out.push(`<h${level}>${inlineMd(match[2])}</h${level}>`);
      i++;
      continue;
    }

    if (RE_RULE.test(line)) { out.push('<hr>'); i++; continue; }

    if (RE_QUOTE.test(line)) {
      const quoted: string[] = [];
      while (i < lines.length && (match = lines[i].match(RE_QUOTE))) { quoted.push(match[1]); i++; }
      out.push(`<blockquote>${renderLines(quoted).join('')}</blockquote>`);
      continue;
    }

    if (isTableStart(lines, i)) {
      const cells = (row: string) => row.trim().replace(/^\||\|$/g, '').split('|').map(cell => cell.trim());
      const head = cells(line);
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && lines[i].trim() && lines[i].includes('|')) { rows.push(cells(lines[i])); i++; }
      out.push(
        '<div class="table-wrap"><table><thead><tr>' + head.map(cell => `<th>${inlineMd(cell)}</th>`).join('') +
        '</tr></thead><tbody>' +
        rows.map(row => '<tr>' + head.map((_, c) => `<td>${inlineMd(row[c] || '')}</td>`).join('') + '</tr>').join('') +
        '</tbody></table></div>',
      );
      continue;
    }

    if (RE_LIST.test(line)) {
      const [html, next] = takeList(lines, i);
      out.push(html);
      i = next;
      continue;
    }

    const paragraph = [line.trim()];
    i++;
    while (i < lines.length && lines[i].trim() && !startsBlock(lines, i)) { paragraph.push(lines[i].trim()); i++; }
    out.push(`<p>${inlineMd(paragraph.join('\n')).replace(/\n/g, '<br>')}</p>`);
  }
  return out;
}

/** Reads one list starting at lines[start]; nested lists and code inside items are rendered recursively. */
function takeList(lines: string[], start: number): [string, number] {
  const first = lines[start].match(RE_LIST)!;
  const base = first[1].length;
  const ordered = /\d/.test(first[2]);
  const isSibling = (line: string, match: RegExpMatchArray | null) =>
    !!match && Math.abs(indentOf(line) - base) <= 1 && /\d/.test(match[2]) === ordered;
  const items: { pad: number; lines: string[] }[] = [];
  let i = start;

  while (i < lines.length) {
    const line = lines[i];
    const match = RE_RULE.test(line) ? null : line.match(RE_LIST);
    const indent = indentOf(line);

    if (match && isSibling(line, match)) {
      items.push({ pad: indent + match[2].length + 1, lines: [match[3]] });
      i++;
    } else if (!line.trim()) {
      let next = i + 1;
      while (next < lines.length && !lines[next].trim()) next++;
      if (next >= lines.length) break;
      const following = lines[next];
      if (!isSibling(following, following.match(RE_LIST)) && indentOf(following) < base + 2) break;
      items[items.length - 1].lines.push('');
      i = next;
    } else if (indent >= base + 2) {
      const item = items[items.length - 1];
      item.lines.push(line.slice(Math.min(indent, item.pad)));
      i++;
    } else if (!match && lines[i - 1].trim() && !startsBlock(lines, i)) {
      items[items.length - 1].lines.push(line.trim());   // text that continues the item without being indented
      i++;
    } else {
      break;
    }
  }

  const body = items.map(item => {
    const blocks = renderLines(item.lines);
    if (blocks[0]?.startsWith('<p>')) blocks[0] = blocks[0].slice(3, -4);
    return `<li>${blocks.join('')}</li>`;
  }).join('');
  const number = parseInt(first[2], 10);
  return [ordered ? `<ol${number > 1 ? ` start="${number}"` : ''}>${body}</ol>` : `<ul>${body}</ul>`, i];
}

type Block = Element & { blockSource?: string };

/** Replaces only the blocks that changed, so earlier text is not rebuilt on every frame while streaming. */
export function patchBlocks(container: Element, blocks: string[]) {
  for (let index = 0; index < blocks.length; index++) {
    const existing = container.children[index] as Block | undefined;
    if (existing && existing.blockSource === blocks[index]) continue;
    const template = document.createElement('template');
    template.innerHTML = blocks[index];
    const node = template.content.firstElementChild as Block;
    node.blockSource = blocks[index];
    if (existing) existing.replaceWith(node); else container.append(node);
  }
  while (container.children.length > blocks.length) container.lastElementChild!.remove();
}

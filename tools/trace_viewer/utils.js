export function summarizeText(value, maxLen) {
  if (!value) {
    return "";
  }
  return value.length > maxLen ? `${value.slice(0, maxLen - 1)}...` : value;
}

export function wrapText(value, maxCharsPerLine, maxLines) {
  if (!value) {
    return [];
  }

  const words = String(value).split(/\s+/).filter(Boolean);
  const lines = [];
  let current = "";

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length <= maxCharsPerLine) {
      current = candidate;
      continue;
    }
    if (current) {
      lines.push(current);
      current = word;
    } else {
      lines.push(`${word.slice(0, Math.max(1, maxCharsPerLine - 3))}...`);
      current = "";
    }
    if (lines.length === maxLines) {
      break;
    }
  }

  if (lines.length < maxLines && current) {
    lines.push(current);
  }

  if (lines.length === maxLines && lines.join(" ").length < String(value).length) {
    lines[maxLines - 1] = `${lines[maxLines - 1].slice(0, Math.max(1, maxCharsPerLine - 3))}...`;
  }

  return lines.slice(0, maxLines);
}

export function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

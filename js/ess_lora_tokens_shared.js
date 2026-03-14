function isNumericWeight(raw) {
  return /^[-+]?(?:\d+\.?\d*|\.\d+)$/.test(String(raw || "").trim());
}

function normalizeLoraName(raw) {
  return String(raw || "")
    .trim()
    .replace(/\\/g, "/")
    .replace(/^\/+/, "")
    .replace(/\/{2,}/g, "/");
}

export function normalizeLoraWeight(raw, fallback = "1") {
  const text = String(raw ?? "").trim();
  if (!text) return String(fallback || "1");
  if (!isNumericWeight(text)) return String(fallback || "1");
  return text;
}

export function parseLoraTokenBody(body) {
  const raw = String(body || "").trim();
  if (!raw) {
    return {
      name: "",
      weight: "",
      displayWeight: "1",
      valid: false,
    };
  }

  const normalized = raw.replace(/\s+/g, " ").trim();
  const parts = normalized.split(":");
  let name = normalized;
  let weight = "";

  if (parts.length >= 2) {
    const possibleWeight = parts[parts.length - 1].trim();
    if (isNumericWeight(possibleWeight)) {
      weight = possibleWeight;
      name = parts.slice(0, -1).join(":").trim();
    }
  }

  name = normalizeLoraName(name);
  const valid = !!name;
  const displayWeight = normalizeLoraWeight(weight || "1");
  return { name, weight, displayWeight, valid };
}

export function normalizeLoraLookupKey(name) {
  const n = normalizeLoraName(name).toLowerCase();
  return n;
}

function matchesKnownLora(knownSet, name) {
  if (!(knownSet instanceof Set) || !name) return true;
  const base = normalizeLoraLookupKey(name);
  if (knownSet.has(base)) return true;
  if (base.endsWith(".safetensors")) {
    return knownSet.has(base.slice(0, -12));
  }
  return knownSet.has(`${base}.safetensors`);
}

export function extractLoraTokens(text, options = {}) {
  const src = String(text || "");
  const knownLoras = options?.knownLoras instanceof Set ? options.knownLoras : null;
  const tokens = [];
  const re = /<lora:([^>\n]+?)>/gi;
  let match;
  while ((match = re.exec(src)) !== null) {
    const rawText = String(match[0] || "");
    const start = Number(match.index || 0);
    const end = start + rawText.length;
    const parsed = parseLoraTokenBody(match[1] || "");
    if (!parsed.valid) {
      tokens.push({
        start,
        end,
        raw: rawText,
        body: String(match[1] || ""),
        name: "",
        weight: "",
        displayWeight: "1",
        valid: false,
        known: false,
      });
      continue;
    }
    const known = matchesKnownLora(knownLoras, parsed.name);
    tokens.push({
      start,
      end,
      raw: rawText,
      body: String(match[1] || ""),
      name: parsed.name,
      weight: parsed.weight,
      displayWeight: parsed.displayWeight,
      valid: true,
      known,
    });
  }
  return tokens;
}

export function findLoraTokenAt(text, pos, options = {}) {
  const index = Number(pos || 0);
  const inclusive = !!options.inclusive;
  const tokens = extractLoraTokens(text, options);
  for (const token of tokens) {
    const inside = inclusive
      ? index >= token.start && index <= token.end
      : index > token.start && index < token.end;
    if (inside) return token;
  }
  return null;
}

export function expandRangeToLoraTokenBoundaries(text, start, end, options = {}) {
  let rangeStart = Math.max(0, Number(start || 0));
  let rangeEnd = Math.max(rangeStart, Number(end || 0));
  const tokens = extractLoraTokens(text, options);
  for (const token of tokens) {
    const intersects = rangeStart < token.end && rangeEnd > token.start;
    if (!intersects) continue;
    rangeStart = Math.min(rangeStart, token.start);
    rangeEnd = Math.max(rangeEnd, token.end);
  }
  return { start: rangeStart, end: rangeEnd };
}

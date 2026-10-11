import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const configPath = process.argv[2];
if (!configPath) throw new Error("Usage: node build_chapter5_master_workbook.mjs <config.json>");
const config = JSON.parse(await fs.readFile(configPath, "utf8"));

function parseCsv(text) {
  text = text.replace(/^\uFEFF/, "");
  const rows = [];
  let row = [], cell = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (c === '"') quoted = false;
      else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(cell); cell = ""; }
    else if (c === '\n') { row.push(cell.replace(/\r$/, "")); rows.push(row); row = []; cell = ""; }
    else cell += c;
  }
  if (cell.length || row.length) { row.push(cell.replace(/\r$/, "")); rows.push(row); }
  return rows.filter(r => !(r.length === 1 && r[0] === ""));
}

function typed(value, header) {
  if (value === "") return null;
  const h = String(header).toLowerCase();
  if (/id|seed|scenario|algorithm|method|network|experiment|paper|status|path|file|hash|role|column|sheet|分区|场景|算法|方法|实验/.test(h)) return value;
  if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) return Number(value);
  return value;
}

function widthFor(values) {
  const maxChars = Math.max(...values.map(v => String(v ?? "").length));
  return Math.max(76, Math.min(300, 14 + maxChars * 7));
}

const wb = Workbook.create();
for (const item of config.sheets) {
  const text = await fs.readFile(item.csv, "utf8");
  const parsed = parseCsv(text);
  if (!parsed.length) throw new Error(`Empty CSV for sheet ${item.name}: ${item.csv}`);
  const headers = parsed[0];
  const matrix = [headers, ...parsed.slice(1).map(r => headers.map((h, i) => typed(r[i] ?? "", h)))];
  const ws = wb.worksheets.add(item.name);
  ws.showGridLines = false;
  ws.freezePanes.freezeRows(1);
  const used = ws.getRangeByIndexes(0, 0, matrix.length, headers.length);
  used.values = matrix;
  ws.getRangeByIndexes(0, 0, 1, headers.length).format = {
    fill: "#155E75",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: "#0E7490" },
  };
  ws.getRangeByIndexes(0, 0, 1, headers.length).format.rowHeightPx = 36;
  if (matrix.length > 1) {
    ws.getRangeByIndexes(1, 0, matrix.length - 1, headers.length).format = {
      borders: { preset: "inside", style: "thin", color: "#E2E8F0" },
      verticalAlignment: "center",
    };
  }
  for (let c = 0; c < headers.length; c++) {
    const col = ws.getRangeByIndexes(0, c, matrix.length, 1);
    col.format.columnWidthPx = widthFor(matrix.map(r => r[c]));
    if (/id|seed|scenario|algorithm|method|network|experiment|paper|status|path|file|hash|role|column|sheet|分区|场景|算法|方法|实验/i.test(headers[c])) {
      col.format.numberFormat = "@";
    } else {
      col.format.numberFormat = "0.###############";
    }
  }
}

const sheetInspection = await wb.inspect({ kind: "sheet", include: "id,name" });
const keyInspection = await wb.inspect({
  kind: "table",
  range: "F5_7_MeanStd!A1:E7",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 8,
});
const errorInspection = await wb.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});

await fs.mkdir(config.previewDir, { recursive: true });
for (const item of config.sheets) {
  // Sample-level CDF sheets contain thousands of rows. Render a bounded,
  // representative viewport for visual QA while retaining every row in XLSX.
  const renderOptions = item.name.startsWith("CDF_")
    ? { sheetName: item.name, range: "A1:J60", scale: 0.7, format: "png" }
    : { sheetName: item.name, autoCrop: "all", scale: 0.7, format: "png" };
  const preview = await wb.render(renderOptions);
  await fs.writeFile(`${config.previewDir}/${item.name}.png`, new Uint8Array(await preview.arrayBuffer()));
}

await fs.writeFile(config.inspectionPath, JSON.stringify({
  sheetCount: config.sheets.length,
  sheetNames: config.sheets.map(s => s.name),
  sheetInspection: sheetInspection.ndjson,
  keyInspection: keyInspection.ndjson,
  errorInspection: errorInspection.ndjson,
}, null, 2), "utf8");

const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(config.outputPath);
process.stdout.write(JSON.stringify({ outputPath: config.outputPath, sheetCount: config.sheets.length, renderedSheets: config.sheets.length }));

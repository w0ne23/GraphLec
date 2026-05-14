import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const xlsxPath = "/Users/cheong-kyumin/Desktop/LLM별 분류표.xlsx";
const jsonPath = "/Users/cheong-kyumin/HSU/캡스톤/graphLec/local_storage/results/d92fe381-188f-4e5e-b613-7aa8d4b09c42/d92fe381-188f-4e5e-b613-7aa8d4b09c42_analyzer/d92fe381-188f-4e5e-b613-7aa8d4b09c42_issue_judge_merged_issue_type_ensemble.json";

const raw = await fs.readFile(jsonPath, "utf8");
const data = JSON.parse(raw);

function summarize(value, depth = 0) {
  if (depth > 3) return typeof value;
  if (Array.isArray(value)) {
    return {
      type: "array",
      length: value.length,
      sample: value.length ? summarize(value[0], depth + 1) : null,
    };
  }
  if (value && typeof value === "object") {
    const keys = Object.keys(value);
    const out = { type: "object", keyCount: keys.length, keys: keys.slice(0, 25) };
    for (const key of keys.slice(0, 8)) out[key] = summarize(value[key], depth + 1);
    return out;
  }
  return { type: typeof value, value };
}

console.log("JSON_SUMMARY");
console.log(JSON.stringify(summarize(data), null, 2));

const input = await FileBlob.load(xlsxPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheets = workbook.worksheets.items.map((sheet) => sheet.name);
console.log("SHEETS");
console.log(JSON.stringify(sheets, null, 2));

for (const sheetName of sheets) {
  const table = await workbook.inspect({
    kind: "table",
    range: `${sheetName}!A1:Z80`,
    include: "values,formulas",
    tableMaxRows: 80,
    tableMaxCols: 26,
  });
  console.log(`TABLE ${sheetName}`);
  console.log(table.ndjson);
}

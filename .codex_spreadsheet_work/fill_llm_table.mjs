import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const xlsxPath = "/Users/cheong-kyumin/Desktop/LLM별 분류표.xlsx";
const jsonPath = "/Users/cheong-kyumin/HSU/캡스톤/graphLec/local_storage/results/d92fe381-188f-4e5e-b613-7aa8d4b09c42/d92fe381-188f-4e5e-b613-7aa8d4b09c42_analyzer/d92fe381-188f-4e5e-b613-7aa8d4b09c42_issue_judge_merged_issue_type_ensemble.json";
const outputDir = "/Users/cheong-kyumin/HSU/캡스톤/graphLec/outputs/d92fe381-llm-classification";
const outputPath = `${outputDir}/LLM별 분류표_채움.xlsx`;

const categories = [
  "factual_error",
  "temporal_error",
  "confusing_explanation",
  "scope_overclaim",
];
const models = ["gpt", "claude", "grok"];

const raw = await fs.readFile(jsonPath, "utf8");
const data = JSON.parse(raw);
const input = await FileBlob.load(xlsxPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.items[0];

for (const [modelIndex, model] of models.entries()) {
  const result = data.model_results?.[model];
  if (!result?.classifications?.length) continue;

  const byId = new Map(result.classifications.map((item) => [item.id, item]));
  const colIndex = 3 + modelIndex; // C, D, E in 1-based Excel columns.

  for (let issueNo = 1; issueNo <= 8; issueNo += 1) {
    const id = `I${String(issueNo).padStart(4, "0")}`;
    const item = byId.get(id);
    if (!item?.probabilities) continue;

    const startRow = 2 + (issueNo - 1) * 4;
    const values = categories.map((category) => [item.probabilities[category] ?? null]);
    sheet.getRangeByIndexes(startRow - 1, colIndex - 1, 4, 1).values = values;
  }
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "formula error scan",
});
console.log("ERROR_SCAN");
console.log(errors.ndjson);

const check = await workbook.inspect({
  kind: "table",
  range: "Sheet1!A1:E33",
  include: "values,formulas",
  tableMaxRows: 33,
  tableMaxCols: 5,
});
console.log("FILLED_TABLE");
console.log(check.ndjson);

await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({ sheetName: "Sheet1", range: "A1:E33", format: "png", scale: 2 });
await fs.writeFile(`${outputDir}/preview_sheet1.png`, Buffer.from(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);

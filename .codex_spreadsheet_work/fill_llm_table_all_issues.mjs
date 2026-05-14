import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const xlsxPath = "/Users/cheong-kyumin/Desktop/LLM별 분류표.xlsx";
const ensembleJsonPath = "/Users/cheong-kyumin/HSU/캡스톤/graphLec/local_storage/results/d92fe381-188f-4e5e-b613-7aa8d4b09c42/d92fe381-188f-4e5e-b613-7aa8d4b09c42_analyzer/d92fe381-188f-4e5e-b613-7aa8d4b09c42_issue_judge_merged_issue_type_ensemble.json";
const outputDir = "/Users/cheong-kyumin/HSU/캡스톤/graphLec/outputs/d92fe381-llm-classification";
const outputPath = `${outputDir}/LLM별 분류표_전체_채움.xlsx`;

const categories = [
  ["factual_error", "사실적"],
  ["temporal_error", "시대적"],
  ["confusing_explanation", "혼동"],
  ["scope_overclaim", "범위"],
];
const models = ["gpt", "claude", "grok"];
const headers = ["", "", "GPT", "Claude", "Grok"];

const data = JSON.parse(await fs.readFile(ensembleJsonPath, "utf8"));
const issueCount = data.summary?.input_issue_count ?? data.classifications?.length ?? 29;
const input = await FileBlob.load(xlsxPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.items[0];

const modelMaps = Object.fromEntries(
  models.map((model) => [
    model,
    new Map((data.model_results?.[model]?.classifications ?? []).map((item) => [item.id, item])),
  ]),
);

const rows = [headers];
for (let issueNo = 1; issueNo <= issueCount; issueNo += 1) {
  const id = `I${String(issueNo).padStart(4, "0")}`;
  for (const [categoryKey, categoryLabel] of categories) {
    const row = [
      categoryKey === "factual_error" ? issueNo : null,
      categoryLabel,
    ];

    for (const model of models) {
      row.push(modelMaps[model].get(id)?.probabilities?.[categoryKey] ?? null);
    }
    rows.push(row);
  }
}

const rowCount = rows.length;
sheet.getRange(`A1:E${rowCount}`).unmerge();
sheet.getRangeByIndexes(0, 0, rowCount, 5).values = rows;

const fullRange = sheet.getRange(`A1:E${rowCount}`);
fullRange.format = {
  font: { name: "Arial", size: 14, color: "#000000" },
  verticalAlignment: "center",
  borders: { preset: "inside", style: "thin", color: "#D9D9D9" },
};
sheet.getRange(`A1:E${rowCount}`).format.borders = {
  preset: "outside",
  style: "thin",
  color: "#D9D9D9",
};
sheet.getRange("A1:E1").format = {
  font: { name: "Arial", size: 16, bold: true, color: "#000000" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: { preset: "outside", style: "thick", color: "#000000" },
};
sheet.getRange(`A2:A${rowCount}`).format.horizontalAlignment = "center";
sheet.getRange(`B2:B${rowCount}`).format.horizontalAlignment = "left";
sheet.getRange(`C2:E${rowCount}`).format.horizontalAlignment = "right";
sheet.getRange(`C2:E${rowCount}`).format.numberFormat = "0.00";
sheet.getRange(`A1:E${rowCount}`).format.rowHeightPx = 42;
sheet.getRange("A:A").format.columnWidthPx = 90;
sheet.getRange("B:B").format.columnWidthPx = 120;
sheet.getRange("C:E").format.columnWidthPx = 115;

for (let issueNo = 1; issueNo <= issueCount; issueNo += 1) {
  const startRow = 2 + (issueNo - 1) * 4;
  const endRow = startRow + 3;
  sheet.getRange(`A${startRow}:A${endRow}`).merge();
  sheet.getRange(`A${startRow}:A${endRow}`).format = {
    horizontalAlignment: "center",
    verticalAlignment: "center",
    font: { name: "Arial", size: 14, color: "#000000" },
  };
  sheet.getRange(`A${startRow}:E${endRow}`).format.borders = {
    preset: "outside",
    style: "thick",
    color: "#000000",
  };
}

const check = await workbook.inspect({
  kind: "table",
  range: `Sheet1!A1:E${rowCount}`,
  include: "values,formulas",
  tableMaxRows: rowCount,
  tableMaxCols: 5,
});
console.log("FILLED_TABLE");
console.log(check.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "formula error scan",
});
console.log("ERROR_SCAN");
console.log(errors.ndjson);

await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({
  sheetName: "Sheet1",
  range: `A1:E${Math.min(rowCount, 60)}`,
  format: "png",
  scale: 2,
});
await fs.writeFile(`${outputDir}/preview_sheet1_all.png`, Buffer.from(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);

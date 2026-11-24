/**
 * Export Service:
 *  - Free: Export .txt
 *  - PRO:  Export formatted Microsoft Word (.docx) using Phetsarath OT font (100% offline)
 */
import * as FileSystem from "expo-file-system";
import * as Sharing from "expo-sharing";
import { Document, Packer, Paragraph, TextRun } from "docx";
import { Buffer } from "buffer";

export async function exportTxt(text: string): Promise<void> {
  const path = `${FileSystem.cacheDirectory}MaeBounmee_OCR_${Date.now()}.txt`;
  await FileSystem.writeAsStringAsync(path, text, { encoding: FileSystem.EncodingType.UTF8 });
  await Sharing.shareAsync(path, { mimeType: "text/plain" });
}

export async function exportDocx(text: string): Promise<void> {
  const paragraphs = text.split(/\r?\n/).map(
    (line) =>
      new Paragraph({
        children: [
          new TextRun({
            text: line,
            font: "Phetsarath OT",
            size: 24 // 12pt
          })
        ]
      })
  );

  const doc = new Document({
    sections: [{ properties: {}, children: paragraphs }]
  });

  const buf = await Packer.toBuffer(doc);
  const b64 = Buffer.from(buf).toString("base64");
  const path = `${FileSystem.cacheDirectory}MaeBounmee_OCR_${Date.now()}.docx`;
  await FileSystem.writeAsStringAsync(path, b64, { encoding: FileSystem.EncodingType.Base64 });
  await Sharing.shareAsync(path, {
    mimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  });
}

import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspace = "/Users/liuxiang/EEG/report/COPA-EEG_数学大会汇报";
const imageDir = path.join(workspace, "build");
const outputPath = path.join(workspace, "COPA-EEG_数学大会汇报.pptx");
const previewDir = path.join(imageDir, "pptx_previews");
const sourcePath = "/Users/liuxiang/EEG/report/COPA-EEG_汇报概述.md";

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(previewDir, { recursive: true });

  const presentation = Presentation.create({
    slideSize: { width: 1280, height: 720 },
  });

  for (let index = 1; index <= 20; index += 1) {
    const imagePath = path.join(
      imageDir,
      `final-slide-${String(index).padStart(2, "0")}.png`,
    );
    const bytes = await fs.readFile(imagePath);
    const slide = presentation.slides.add();
    slide.background.fill = "#FFFFFF";
    slide.images.add({
      blob: bytes,
      contentType: "image/png",
      alt: `COPA-EEG 数学大会汇报第 ${index} 页`,
      fit: "cover",
      position: { left: 0, top: 0, width: 1280, height: 720 },
    });
    slide.speakerNotes.textFrame.setText(
      `[Sources]\n- ${sourcePath}\n[/Sources]`,
    );

    const preview = await presentation.export({
      slide,
      format: "png",
      scale: 1,
    });
    await writeBlob(
      path.join(
        previewDir,
        `slide-${String(index).padStart(2, "0")}.png`,
      ),
      preview,
    );
  }

  const montage = await presentation.export({
    format: "webp",
    montage: true,
    scale: 1,
  });
  await writeBlob(path.join(imageDir, "pptx-montage.webp"), montage);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(outputPath);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

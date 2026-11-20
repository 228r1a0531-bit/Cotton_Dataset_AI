/* Exercise the headless Streamlit dashboard using Playwright and Chromium. */
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const { chromium } = require("playwright");

const root = process.env.COTTONLEAF_TEST_URL || "http://127.0.0.1:8519";
const routes = [
  ["", "CottonLeaf AI", "Overview"],
  ["dataset", "Dataset & Training"],
  ["inspect_image", "Upload & Classification"],
  ["comparisons", "Model Comparisons"],
  ["evaluation", "Evaluation Metrics"],
  ["architecture", "Model Architecture"],
];

async function main() {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined,
    args: process.env.CHROMIUM_PATH ? ["--no-sandbox"] : [],
  });
  try {
    const page = await browser.newPage({ acceptDownloads: true, viewport: { width: 1440, height: 960 } });
    const pageErrors = [];
    const failedLocalRequests = [];
    const fallbackEndpoints = new Set();
    const reachableEndpoints = new Set();
    page.on("pageerror", (error) => pageErrors.push(error.message));
    page.on("response", (response) => {
      if (!response.url().startsWith(root + "/")) return;
      const path = response.url().slice(root.length);
      const internal = path.match(/^\/([^/]+)\/_stcore\/(health|host-config)$/);
      const topLevel = path.match(/^\/_stcore\/(health|host-config)$/);
      if (response.ok() && topLevel) reachableEndpoints.add(topLevel[1]);
      if (response.status() === 404 && internal && routes.some(([route]) => route === internal[1])) {
        // Streamlit probes the deep-link directory before retrying at the app root.
        fallbackEndpoints.add(internal[2]);
      } else if (response.status() >= 400) {
        failedLocalRequests.push(response.status() + " " + response.url());
      }
    });
    const heading = (name) => page.getByRole("heading", { level: 1, name, exact: true });
    const nav = (route) => page.locator('[data-testid="stSidebarNavLink"][href$="/' + route + '"]');
    const noOverflow = async () => {
      assert.ok(await page.locator('[data-testid="stMain"]').evaluate((main) => (
        main.scrollWidth <= main.clientWidth + 2
      )), "Horizontal overflow in Streamlit content");
      assert.equal(await page.getByTestId("stException").count(), 0);
    };

    await page.goto(root);
    await heading("CottonLeaf AI").waitFor();
    await page.getByText("99.41%", { exact: true }).waitFor();
    await page.getByText("Fusarium wilt", { exact: true }).waitFor();
    if (process.env.COTTONLEAF_SCREENSHOT) {
      await page.evaluate(() => document.fonts.ready);
      await page.screenshot({ path: process.env.COTTONLEAF_SCREENSHOT });
    }
    for (const [route, title] of routes.slice(1)) {
      await nav(route).click();
      await heading(title).waitFor();
      await noOverflow();
    }
    await nav("").click();
    await heading("CottonLeaf AI").waitFor();

    await nav("comparisons").click();
    const published = page.getByRole("tabpanel", { name: "Published Research" });
    const charts = published.getByTestId("stPlotlyChart");
    await published.getByText("Published research results — not reproduced locally.", { exact: true }).waitFor();
    const chart = charts.first();
    await chart.locator(".barlayer path").first().waitFor();
    assert.equal(await chart.locator(".barlayer path").count(), 4);
    await chart.hover();
    const [png] = await Promise.all([
      page.waitForEvent("download"),
      chart.locator('[data-title="Download plot as a PNG"]').click(),
    ]);
    assert.deepEqual((await fs.readFile(await png.path())).subarray(0, 8),
      Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));

    const [csv] = await Promise.all([
      page.waitForEvent("download"),
      published.locator(".st-key-csv_published_benchmarks").getByRole("button", { name: "Download CSV" }).click(),
    ]);
    const values = await fs.readFile(await csv.path(), "utf8");
    assert.ok(values.includes("Feature extraction,CN2-SE + Modified InceptionV3,Accuracy,98.83"));
    assert.ok(values.includes("Final classification,Hybrid + MLP,Accuracy,99.41"));
    await page.getByRole("tab", { name: "Our Experiment" }).click();
    await page.getByText("Published and local metric provenance").waitFor();

    await nav("evaluation").click();
    for (const name of ["Confusion matrix", "ROC / AUC", "Five-fold cross-validation"]) {
      await page.getByRole("tab", { name, exact: true }).click();
      await page.getByRole("tabpanel", { name, exact: true }).waitFor();
    }
    await page.getByText("Published reference values", { exact: true }).waitFor();
    const folds = page.getByRole("tabpanel", { name: "Five-fold cross-validation" });
    await folds.locator(".barlayer path").first().waitFor();
    assert.equal(await folds.locator(".barlayer path").count(), 5);

    await nav("architecture").click();
    for (const [name, label] of [
      ["Hybrid feature fusion", "Four-class softmax"],
      ["Squeeze-and-excitation", "Dense / sigmoid"],
      ["Modified stem", "Feature maps"],
    ]) {
      await page.getByRole("tab", { name }).click();
      await page.getByRole("tabpanel", { name }).getByText(label, { exact: true }).waitFor();
    }

    await nav("inspect_image").click();
    const input = page.locator('input[type="file"]');
    await input.setInputFiles({ name: "damaged.png", mimeType: "image/png", buffer: Buffer.from("bad image") });
    await page.getByText(/Could not open the image/).waitFor();
    for (const file of [process.env.COTTONLEAF_TEST_PNG, process.env.COTTONLEAF_TEST_JPG].filter(Boolean)) {
      await input.setInputFiles(file);
      await page.getByText(file.split(/[\\/]/).pop(), { exact: true }).waitFor();
      await page.getByTestId("stImage").locator("img").waitFor();
      await page.waitForFunction(() => [...document.querySelectorAll('[data-testid="stImage"] img')]
        .some((image) => image.naturalWidth > 0));
    }
    if (process.env.COTTONLEAF_EXPECT_CHECKPOINT) {
      await page.getByRole("button", { name: "Classify image" }).click();
      await page.getByText("Inference complete.", { exact: true }).waitFor();
      await page.getByText(/^Checkpoint model:/).waitFor();
      await page.getByTestId("stPlotlyChart").waitFor();
    } else {
      await page.getByText(/Inference unavailable/).waitFor();
      assert.ok(await page.getByRole("button", { name: "Classify image" }).isDisabled());
    }

    for (const width of [1024, 390]) {
      await page.setViewportSize({ width, height: 850 });
      for (const [route, title] of routes) {
        await page.goto(root + "/" + route);
        await heading(title).waitFor();
        await noOverflow();
      }
    }
    await page.goto(root);
    await heading("CottonLeaf AI").waitFor();
    await page.getByTestId("stExpandSidebarButton").click();
    await nav("inspect_image").click();
    await heading("Upload & Classification").waitFor();
    await noOverflow();
    for (const endpoint of fallbackEndpoints) assert.ok(reachableEndpoints.has(endpoint), endpoint);
    assert.deepEqual(pageErrors, []);
    assert.deepEqual(failedLocalRequests, []);
    console.log("Navigation, downloads, upload, inference state, and 1440/1024/390 px browser checks passed.");
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

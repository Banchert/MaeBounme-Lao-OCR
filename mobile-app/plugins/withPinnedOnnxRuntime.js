const fs = require("fs");
const path = require("path");
const { withDangerousMod, withAppBuildGradle } = require("@expo/config-plugins");

/**
 * 1) Pins onnxruntime-android from `latest.integration` to the exact installed
 *    `onnxruntime-react-native` package version so C++ headers match 100% AND
 *    Expo EAS Maven cache proxy resolves the AAR cleanly.
 * 2) Adds packagingOptions pickFirst for libreactnative.so / libjsi.so / libfbjni.so / libc++_shared.so
 *    so :app:mergeReleaseNativeLibs succeeds cleanly on RN 0.76 + onnxruntime-react-native.
 */
module.exports = function withPinnedOnnxRuntime(config) {
  config = withDangerousMod(config, [
    "android",
    async (cfg) => {
      const ortRoot = path.join(cfg.modRequest.projectRoot, "node_modules", "onnxruntime-react-native");
      const pkgPath = path.join(ortRoot, "package.json");
      const gradlePath = path.join(ortRoot, "android", "build.gradle");
      if (fs.existsSync(pkgPath) && fs.existsSync(gradlePath)) {
        const ortVersion = JSON.parse(fs.readFileSync(pkgPath, "utf8")).version;
        let content = fs.readFileSync(gradlePath, "utf8");
        content = content.replace(/latest\.integration/g, ortVersion);
        fs.writeFileSync(gradlePath, content, "utf8");
      }
      return cfg;
    }
  ]);

  config = withAppBuildGradle(config, (cfg) => {
    const snippet = `
    packagingOptions {
        pickFirst '**/libreactnative.so'
        pickFirst '**/libjsi.so'
        pickFirst '**/libfbjni.so'
        pickFirst '**/libc++_shared.so'
    }
`;
    if (!cfg.modResults.contents.includes("libreactnative.so")) {
      cfg.modResults.contents = cfg.modResults.contents.replace(
        /android\s*\{/,
        `android {${snippet}`
      );
    }
    return cfg;
  });

  return config;
};

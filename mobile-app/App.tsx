import React, { useEffect, useState } from "react";
import { StatusBar } from "expo-status-bar";
import * as Font from "expo-font";
import HomeScreen from "./src/screens/HomeScreen";
import ResultScreen from "./src/screens/ResultScreen";
import { UiLang } from "./src/i18n/translations";

export default function App() {
  const [uiLang, setUiLang] = useState<UiLang>("lo");
  const [result, setResult] = useState<{
    text: string;
    ms: number;
    avgConf: number;
    lineCount: number;
    images: string[];
    rotations: number[];
  } | null>(null);

  useEffect(() => {
    Font.loadAsync({
      PhetsarathOT: require("./assets/fonts/PhetsarathOT.ttf")
    }).catch(() => {});
  }, []);

  return (
    <>
      <StatusBar style="light" />
      {result ? (
        <ResultScreen
          uiLang={uiLang}
          initialText={result.text}
          ms={result.ms}
          avgConf={result.avgConf}
          lineCount={result.lineCount}
          images={result.images}
          rotations={result.rotations}
          onBack={() => setResult(null)}
        />
      ) : (
        <HomeScreen
          uiLang={uiLang}
          setUiLang={setUiLang}
          onResult={(text, ms, avgConf, lineCount, images, rotations) =>
            setResult({ text, ms, avgConf, lineCount, images, rotations })
          }
        />
      )}
    </>
  );
}

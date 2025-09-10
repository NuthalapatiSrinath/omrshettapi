// src/pages/Results/Results.jsx
import React, { useState, useMemo } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import * as XLSX from "xlsx";
import styles from "./Results.module.css";

/**
 * Results page:
 * - Accepts extracted OMR JSON via location.state.extracted
 * - Allows uploading an answer-key file (.xlsx, .xls, .csv)
 * - Compares and displays results
 */

function parseKeyFromWorkbook(workbook) {
  // Get first sheet
  const sheetName = workbook.SheetNames[0];
  const sheet = workbook.Sheets[sheetName];

  // Try to convert to JSON using header detection
  const rawJson = XLSX.utils.sheet_to_json(sheet, { header: 1 });

  // rawJson is an array of rows, each row is array of cells.
  // We'll try to detect simple formats:
  // 1) rows like [question, answer]  => map q->answer
  // 2) one row or one column of answers => map by index (1-based)
  // 3) if header row present (strings like 'q' or 'question'), skip header

  if (!rawJson || rawJson.length === 0) return {};

  // Trim empty rows
  const rows = rawJson.filter(
    (r) =>
      r &&
      r.some((c) => c !== undefined && c !== null && String(c).trim() !== "")
  );

  if (rows.length === 0) return {};

  // Case A: first non-empty row looks like headers ("q","answer" or "question","answer")
  const firstRow = rows[0].map((c) =>
    c === undefined || c === null ? "" : String(c).trim().toLowerCase()
  );
  const hasHeader =
    firstRow.length >= 2 &&
    (firstRow[0].includes("q") ||
      firstRow[0].includes("question") ||
      firstRow[1].includes("answer"));

  const map = {};

  if (hasHeader) {
    // interpret following rows as [q,answer]
    for (let i = 1; i < rows.length; i++) {
      const row = rows[i];
      if (!row || row.length === 0) continue;
      const qCell = row[0];
      const aCell = row[1];
      const q = Number(qCell);
      if (
        !Number.isNaN(q) &&
        aCell !== undefined &&
        aCell !== null &&
        String(aCell).trim() !== ""
      ) {
        map[q] = String(aCell).trim().toUpperCase();
      }
    }
    return map;
  }

  // Case B: rows look like [q, answer] directly (no header) — detect if many rows have two columns and first is numeric
  const rowsWithTwoCols = rows.filter(
    (r) => r.length >= 2 && !isNaN(Number(r[0]))
  );
  if (rowsWithTwoCols.length >= Math.floor(rows.length * 0.5)) {
    for (const row of rowsWithTwoCols) {
      const q = Number(row[0]);
      const a = row[1];
      if (
        !Number.isNaN(q) &&
        a !== undefined &&
        a !== null &&
        String(a).trim() !== ""
      ) {
        map[q] = String(a).trim().toUpperCase();
      }
    }
    return map;
  }

  // Case C: single row of answers e.g. [A,B,C,D,...]
  if (rows.length === 1 && rows[0].length >= 1) {
    rows[0].forEach((cell, idx) => {
      const ans =
        cell === undefined || cell === null
          ? ""
          : String(cell).trim().toUpperCase();
      if (ans) map[idx + 1] = ans; // question numbers start at 1
    });
    return map;
  }

  // Case D: single column of answers: [[A],[B],[C],...]
  if (rows.length >= 1 && rows.every((r) => r.length === 1)) {
    rows.forEach((r, idx) => {
      const ans =
        r[0] === undefined || r[0] === null
          ? ""
          : String(r[0]).trim().toUpperCase();
      if (ans) map[idx + 1] = ans;
    });
    return map;
  }

  // Last resort: try to read rows that look like [something, letter]
  for (const row of rows) {
    if (row.length >= 2) {
      const maybeQ = Number(row[0]);
      const maybeA = row[1];
      if (
        !Number.isNaN(maybeQ) &&
        maybeA !== undefined &&
        String(maybeA).trim() !== ""
      ) {
        map[maybeQ] = String(maybeA).trim().toUpperCase();
      }
    }
  }

  return map;
}

export default function Results() {
  const { state } = useLocation();
  const navigate = useNavigate();
  const extracted = state?.extracted || null; // extracted must be passed when navigating here

  const [keyMap, setKeyMap] = useState(null); // { qnum: 'A', ... }
  const [error, setError] = useState(null);

  // results: computed from extracted and keyMap
  const results = useMemo(() => {
    if (!extracted || !Array.isArray(extracted.questions)) return null;
    const qArr = extracted.questions;

    if (!keyMap) {
      // we can still present extracted answers without comparison
      return {
        summary: null,
        rows: qArr.map((q) => ({
          q: q.q,
          student: q.selected || null,
          correct: null,
          status: "no-key",
        })),
      };
    }

    let correct = 0,
      wrong = 0,
      unanswered = 0;
    const rows = qArr.map((q) => {
      const qnum = q.q;
      const stud = q.selected || null;
      const key = keyMap[qnum] || null;
      let status = "unanswered";
      if (!stud) {
        unanswered++;
        status = "unanswered";
      } else if (!key) {
        status = "no-key";
      } else if (stud === key) {
        correct++;
        status = "correct";
      } else {
        wrong++;
        status = "wrong";
      }
      return { q: qnum, student: stud, correct: key, status };
    });
    const total = rows.length;
    const pct = total ? Math.round((correct / total) * 10000) / 100 : 0;
    return {
      summary: { total, correct, wrong, unanswered, pct },
      rows,
    };
  }, [extracted, keyMap]);

  if (!extracted) {
    return (
      <div className={styles.container}>
        <h2>No extracted OMR data</h2>
        <p>
          Please upload an OMR sheet first (or navigate here with extracted
          JSON).
        </p>
        <button onClick={() => navigate("/dashboard")}>Go back</button>
      </div>
    );
  }

  const handleKeyFile = async (file) => {
    setError(null);
    if (!file) return;
    const name = (file.name || "").toLowerCase();
    try {
      const data = await file.arrayBuffer();
      const workbook = XLSX.read(data, { type: "array" });
      const map = parseKeyFromWorkbook(workbook);
      if (!map || Object.keys(map).length === 0) {
        setError(
          "Could not parse key file. Please ensure it has Q/A pairs or a single row/column of answers."
        );
        setKeyMap(null);
        return;
      }
      console.log("[Results] parsed key map:", map);
      setKeyMap(map);
    } catch (err) {
      console.error("Failed to parse key file", err);
      setError("Failed to parse key file. See console for details.");
      setKeyMap(null);
    }
  };

  const handleFileInput = (ev) => {
    const f = ev.target.files && ev.target.files[0];
    handleKeyFile(f);
  };

  const downloadComparison = () => {
    if (!results) return;
    const rows = results.rows.map((r) => ({
      Question: r.q,
      Student: r.student || "",
      Key: r.correct || "",
      Status: r.status,
    }));
    const ws = XLSX.utils.json_to_sheet(rows);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "results");
    const wbout = XLSX.write(wb, { bookType: "xlsx", type: "array" });
    const blob = new Blob([wbout], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "omr_results.xlsx";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className={styles.container}>
      <h2>OMR Results</h2>

      <div className={styles.topRow}>
        <div>
          <p>
            <strong>Roll:</strong> {extracted.roll || "N/A"}
          </p>
          <p>
            <strong>Detected questions:</strong>{" "}
            {extracted.questions?.length ?? 0}
          </p>
        </div>

        <div className={styles.keyUpload}>
          <label className={styles.fileLabel}>
            Upload Answer Key (xlsx / csv)
            <input
              type="file"
              accept=".xlsx,.xls,.csv"
              onChange={handleFileInput}
            />
          </label>
          <div style={{ marginTop: 8 }}>
            <button
              onClick={() => {
                setKeyMap(null);
                setError(null);
              }}
            >
              Clear Key
            </button>
            <button
              onClick={downloadComparison}
              disabled={!results || !results.summary}
            >
              Download Results
            </button>
          </div>
          {error && <div className={styles.error}>{error}</div>}
        </div>
      </div>

      <div className={styles.summary}>
        {results && results.summary ? (
          <>
            <div>
              <strong>Total:</strong> {results.summary.total}
            </div>
            <div>
              <strong>Correct:</strong> {results.summary.correct}
            </div>
            <div>
              <strong>Wrong:</strong> {results.summary.wrong}
            </div>
            <div>
              <strong>Unanswered:</strong> {results.summary.unanswered}
            </div>
            <div>
              <strong>Score:</strong> {results.summary.pct}%
            </div>
          </>
        ) : (
          <div>No key uploaded yet — showing detected answers only.</div>
        )}
      </div>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Q</th>
              <th>Student</th>
              <th>Key</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {(results?.rows || []).map((r) => (
              <tr key={r.q} className={styles[r.status] || ""}>
                <td>{r.q}</td>
                <td>{r.student || "-"}</td>
                <td>{r.correct || "-"}</td>
                <td>{r.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// src/pages/UploadOmr/UploadOmr.jsx
import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import API from "../../api"; // axios instance with baseURL
import Button from "../../components/Button/Button";
import { toast } from "react-toastify";

export default function UploadOmr() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [extracted, setExtracted] = useState(null);
  const [error, setError] = useState(null);

  const inputRef = useRef();
  const navigate = useNavigate();

  // Handle file selection
  const handleFileChange = (e) => {
    setError(null);
    setExtracted(null);
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);

    if (f.type.startsWith("image/")) {
      const url = URL.createObjectURL(f);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null);
    }
  };

  const handleRemove = () => {
    setFile(null);
    setPreviewUrl(null);
    setExtracted(null);
    setError(null);
    setProgress(0);
    if (inputRef.current) inputRef.current.value = null;
  };

  // Upload to backend
  const handleUpload = async () => {
    if (!file) {
      toast.error("Please choose a file to upload");
      return;
    }

    setUploading(true);
    setProgress(0);
    setError(null);
    setExtracted(null);

    try {
      const formData = new FormData();
      formData.append("omrSheet", file, file.name);

      console.log("[UploadOmr] uploading file:", file);

      const response = await API.post("/omrReader/readOMR", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percent = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setProgress(percent);
          }
        },
      });

      console.log("[UploadOmr] server response:", response);

      const payload = response?.data;
      if (!payload) throw new Error("Empty response from server");

      if (payload.success) {
        // Backend with Python script returns { success, roll, questions }
        setExtracted(payload);
        toast.success("Text extracted successfully");
      } else {
        const msg = payload.message || "Text extraction failed";
        setError(msg);
        toast.error(msg);
      }
    } catch (err) {
      console.error("[UploadOmr] error:", err);
      const serverMsg =
        err?.response?.data?.message || err?.message || "Upload failed";
      setError(serverMsg);
      toast.error(serverMsg);
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  const goToResults = () => {
    if (!extracted) {
      toast.error("No extracted data available");
      return;
    }
    navigate("/results", { state: { extracted } });
  };

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: 16 }}>
      <h2>Upload OMR Sheet</h2>

      <div
        style={{
          display: "flex",
          gap: 20,
          alignItems: "flex-start",
          flexWrap: "wrap",
        }}
      >
        <div style={{ flex: 1, minWidth: 280 }}>
          <input
            ref={inputRef}
            type="file"
            accept="image/*,application/pdf"
            onChange={handleFileChange}
            style={{ display: "block", marginBottom: 12 }}
          />

          {previewUrl && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 13, marginBottom: 6 }}>Preview</div>
              <img
                src={previewUrl}
                alt="preview"
                style={{
                  maxWidth: "100%",
                  height: "auto",
                  borderRadius: 8,
                  boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
                }}
              />
            </div>
          )}

          <div style={{ display: "flex", gap: 8 }}>
            <Button
              type="secondary"
              label="Remove"
              onClick={handleRemove}
              disabled={!file || uploading}
            />
            <Button
              type="primary"
              label={
                uploading ? `Uploading... ${progress}%` : "Upload & Extract"
              }
              onClick={handleUpload}
              disabled={!file || uploading}
            />
          </div>

          {uploading && (
            <div style={{ marginTop: 10 }}>
              <div
                style={{
                  height: 8,
                  background: "#eee",
                  borderRadius: 8,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${progress}%`,
                    height: "100%",
                    background: "#6A4CF6",
                  }}
                />
              </div>
            </div>
          )}

          {error && (
            <div style={{ marginTop: 12, color: "crimson" }}>
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>

        <div style={{ flex: 1, minWidth: 300 }}>
          <h4>Extracted Data</h4>
          {!extracted ? (
            <div style={{ color: "#666" }}>No extracted text yet.</div>
          ) : (
            <div
              style={{
                background: "#fafafa",
                padding: 12,
                borderRadius: 8,
                maxHeight: 480,
                overflow: "auto",
                fontFamily: "monospace",
                fontSize: 13,
              }}
            >
              <div>
                <strong>Roll:</strong> {extracted.roll || "N/A"}
              </div>
              <ul>
                {extracted.questions?.map((q) => (
                  <li key={q.q}>
                    Q{q.q}: {q.selected || "-"}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {extracted && (
            <div style={{ marginTop: 12 }}>
              <Button
                type="primary"
                label="Go to Results"
                onClick={goToResults}
              />
            </div>
          )}
        </div>
      </div>

      <div style={{ marginTop: 18, fontSize: 13, color: "#666" }}>
        Note: Uploads the file to your backend which runs
        Python/OpenCV/Tesseract to extract data from the OMR sheet.
      </div>
    </div>
  );
}

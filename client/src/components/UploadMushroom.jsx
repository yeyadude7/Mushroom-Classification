import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";

function UploadMushroom() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Predict
  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Upload error:", err);
    } finally {
      setLoading(false);
    }
  };

  // Grad-CAM
  const handleVisualize = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch("http://127.0.0.1:8000/visualize", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult((prev) => ({ ...prev, gradcam: data.gradcam_image }));
    } catch (err) {
      console.error("Grad-CAM error:", err);
    }
  };

  // Reset
  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  const resultColor =
    result?.class === "poisonous" ? "danger" : result ? "success" : "secondary";

  return (
    <div className="container py-5">
      <div className="row justify-content-center">
        <div className="col-lg-8 col-md-10">
          <div className="card shadow border-0 p-4 animate__animated animate__fadeIn">
            <h2 className="text-center mb-3 fw-bold">
              🍄 Mushroom Classifier
            </h2>
            <p className="text-muted text-center mb-4">
              Upload a mushroom image to identify whether it’s edible or poisonous.
            </p>

            {/* Upload */}
            <form onSubmit={handleUpload} className="text-center">
              <div className="mb-3">
                <input
                  type="file"
                  accept="image/*"
                  className="form-control form-control-md"
                  onChange={(e) => {
                    const f = e.target.files[0];
                    setFile(f);
                    setPreview(URL.createObjectURL(f));
                  }}
                />
              </div>

              <div className="d-flex justify-content-center gap-3 flex-wrap">
                <button
                  type="submit"
                  className="btn btn-primary px-4"
                  disabled={!file || loading}
                >
                  {loading ? (
                    <>
                      <span
                        className="spinner-border spinner-border-sm me-2"
                        role="status"
                      ></span>
                      Predicting…
                    </>
                  ) : (
                    "Upload & Predict"
                  )}
                </button>

                {result && (
                  <>
                    <button
                      type="button"
                      className="btn btn-outline-success px-4"
                      onClick={handleVisualize}
                    >
                      Grad-CAM Heatmap
                    </button>
                    <button
                      type="button"
                      className="btn btn-outline-secondary px-4"
                      onClick={handleReset}
                    >
                      Reset
                    </button>
                  </>
                )}
              </div>
            </form>

            {/* Preview */}
            {preview && (
              <div className="text-center mt-4">
                <h5 className="fw-semibold mb-2">Image Preview</h5>
                <img
                  src={preview}
                  alt="Uploaded preview"
                  className="img-fluid rounded shadow-sm border"
                  style={{ maxHeight: "320px", objectFit: "cover" }}
                />
              </div>
            )}

            {/* Results */}
            {result && (
            <div className={`card bg-${resultColor}-subtle mt-4 border-${resultColor}`}>
                <div className="card-body">
                <h5
                    className={`card-title text-${resultColor} fw-bold text-center`}
                >
                    {result.class.toUpperCase()}
                </h5>
                <p className="text-center text-muted mb-3">Confidence Score</p>

                <div className="progress mb-3" style={{ height: "1rem" }}>
                    <div
                    className={`progress-bar bg-${resultColor}`}
                    style={{ width: `${result.confidence * 100}%` }}
                    role="progressbar"
                    ></div>
                </div>
                <p className="text-center fw-medium mb-0">
                    {(result.confidence * 100).toFixed(2)} %
                </p>

                {result.probabilities && (
                    <div className="mt-3 text-center small">
                    <p className="mb-1">
                        <strong>Edible:</strong>{" "}
                        {(result.probabilities.edible * 100).toFixed(2)} %
                    </p>
                    <p className="mb-0">
                        <strong>Poisonous:</strong>{" "}
                        {(result.probabilities.poisonous * 100).toFixed(2)} %
                    </p>
                    </div>
                )}

                {/* ⚙️ Metrics Footer */}
                <div className="mt-4 border-top pt-3 small text-muted">
                    <div className="row text-center">
                    <div className="col-sm-4 mb-2">
                        <strong>Entropy:</strong>
                        <br />
                        {result.entropy?.toFixed(4)}
                    </div>
                    <div className="col-sm-4 mb-2">
                        <strong>Margin:</strong>
                        <br />
                        {result.margin?.toFixed(4)}
                    </div>
                    <div className="col-sm-4 mb-2">
                        <strong>Inference Time:</strong>
                        <br />
                        {result.inference_time_ms} ms
                    </div>
                    </div>
                </div>
                </div>
            </div>
            )}


            {/* Grad-CAM */}
            {result?.gradcam && (
              <div className="text-center mt-5">
                <h5 className="fw-semibold mb-3">
                  Model Attention Heatmap
                </h5>
                <img
                  src={`data:image/jpeg;base64,${result.gradcam}`}
                  alt="Grad-CAM Heatmap"
                  className="img-fluid rounded shadow border"
                  style={{ maxHeight: "420px" }}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default UploadMushroom;

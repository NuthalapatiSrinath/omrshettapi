import { useState } from "react";
import styles from "./GenerateOmr.module.css";
import Button from "./../../components/Button/Button";
import { motion } from "framer-motion";
import { closeModal } from "../../redux/slices/modalSlice";
import { useDispatch } from "react-redux";
import InputBox from "./../../components/InputBox/InputBox";
import { createOMR } from "../../api/services/omrService";
import { toast } from "react-toastify";

// GenerateOmr with optimistic UI dispatches:
// - dispatches {detail: tempObj} on 'omrCreated' immediately
// - when server returns created object, dispatches {detail: { tempId, created }} on 'omrReplace'
// - on failure, dispatches {detail: { tempId }} on 'omrCreateFailed'
const GenerateOmr = () => {
  const dispatch = useDispatch();
  const [questions, setQuestions] = useState("");
  const [options, setOptions] = useState("");
  const [integers, setIntegers] = useState("");
  const [formErrors, setFormErrors] = useState({});
  const [loading, setLoading] = useState(false);

  const handleAdd = async () => {
    setFormErrors({});

    // Basic validation
    const errs = {};
    if (!questions || Number(questions) <= 0) errs.numberOfQuestions = true;
    if (!options || Number(options) < 2) errs.numberOfOptions = true;
    if (integers === "" || Number(integers) < 0)
      errs.numberOfIntegerQuestions = true;

    if (Object.keys(errs).length) {
      setFormErrors(errs);
      toast.error("Please fill required fields correctly");
      return;
    }

    // Create a temporary optimistic object
    const tempId = `temp-${Date.now()}`;
    const author =
      JSON.parse(localStorage.getItem("user") || "null") ||
      localStorage.getItem("user") ||
      "default";
    const tempObj = {
      _id: tempId,
      numberOfQuestions: Number(questions),
      numberOfOptions: Number(options),
      numberOfIntegerQuestions: Number(integers || 0),
      author: typeof author === "string" ? author : "default",
      // optional UI fields to help display card similarly
      createdAt: new Date().toISOString(),
      // you can include other fields expected by OMRCard (e.g., title) if needed
    };

    // Immediately show optimistic card
    window.dispatchEvent(new CustomEvent("omrCreated", { detail: tempObj }));

    setLoading(true);

    try {
      // Call the API
      const response = await createOMR(
        Number(questions),
        Number(options),
        Number(integers || 0),
        typeof author === "string" ? author : "default"
      );

      // Normalize created object from response
      // Your service returns response.data; try common shapes
      let created = null;
      if (!response) {
        created = null;
      } else if (response._id || response.id) {
        // created doc returned directly
        created = response;
      } else if (response.data && (response.data._id || response.data.id)) {
        created = response.data;
      } else if (
        response.payload &&
        response.payload.data &&
        (response.payload.data._id || response.payload.data.id)
      ) {
        created = response.payload.data;
      } else if (
        response.data &&
        Array.isArray(response.data) &&
        response.data.length === 1
      ) {
        created = response.data[0];
      } else {
        // fallback: response maybe wrapped - inspect `response`
        created = response;
      }

      // If we have created object with real _id, replace temp
      if (created && created._id) {
        window.dispatchEvent(
          new CustomEvent("omrReplace", { detail: { tempId, created } })
        );
        toast.success(response?.message || "OMR created successfully");
        dispatch(closeModal());
      } else {
        // If server didn't send created doc, try to fetch single newest item (server fallback)
        // We'll dispatch an event with no created detail so Dashboard will re-fetch — but better to attempt minimal approach.
        // For simplicity here, we'll dispatch omrReplace with created=null to trigger Dashboard re-fetch (handled below).
        window.dispatchEvent(
          new CustomEvent("omrReplace", { detail: { tempId, created: null } })
        );
        toast.success(
          response?.message || "OMR created (server did not return object)."
        );
        dispatch(closeModal());
      }
    } catch (error) {
      console.error("createOMR failed:", error);
      // Remove optimistic card
      window.dispatchEvent(
        new CustomEvent("omrCreateFailed", { detail: { tempId } })
      );
      // Show server validation messages if present
      const errorList = error?.response?.data?.errors;
      if (Array.isArray(errorList)) {
        const errorObj = {};
        errorList.forEach(({ name, message }) => {
          errorObj[name] = true;
          toast.error(`${message}`);
        });
        setFormErrors(errorObj);
      } else {
        toast.error(error?.response?.data?.message || "Something went wrong");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    dispatch(closeModal());
  };

  return (
    <motion.div
      className={styles.modal}
      initial={{ x: "100%", y: "-50%", opacity: 0 }}
      animate={{ x: "-50%", y: "-50%", opacity: 1 }}
      transition={{ type: "spring", stiffness: 120, damping: 15 }}
      style={{
        position: "fixed",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        zIndex: 1000,
      }}
    >
      <h3 className={styles.title}>Generate OMR</h3>

      <InputBox
        type="number"
        label="No.of Questions"
        placeholder="Enter no.of questions"
        value={questions}
        onChange={setQuestions}
        isError={!!formErrors.numberOfQuestions}
      />

      <InputBox
        type="number"
        label="No.of Options"
        placeholder="Enter no.of options"
        value={options}
        onChange={setOptions}
        isError={!!formErrors.numberOfOptions}
      />

      <InputBox
        type="number"
        label="No.of Integer Questions"
        placeholder="Enter no.of integers"
        value={integers}
        onChange={setIntegers}
        isError={!!formErrors.numberOfIntegerQuestions}
      />

      <div className={styles.actions}>
        <Button type="secondary" label="Cancel" onClick={handleCancel} />
        <Button
          type="primary"
          label={loading ? "ADDING..." : "ADD"}
          onClick={handleAdd}
        />
      </div>
    </motion.div>
  );
};

export default GenerateOmr;

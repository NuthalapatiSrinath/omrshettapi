import { useEffect, useState } from "react";
import styles from "./Dashboard.module.css";
import OMRCard from "../../components/OMRCard/OMRCard";
import SearchBar from "../../components/SearchBar/SearchBar";
import IconAndName from "../../components/IconAndName/IconAndName";
import { PiExamFill } from "react-icons/pi";
import Button from "../../components/Button/Button";

import { useDispatch } from "react-redux";
import omrImage1 from "../../assets/images/omrcard.png"; // same image used
import { openModal } from "../../redux/slices/modalSlice";
import { fetchOMRSheets } from "../../api/services/omrService";
import {
  filterDefaultOMR,
  filterUserGeneratedOMR,
} from "../../utils/OMRDataUtils";
import { useNavigate } from "react-router";

const Dashboard = () => {
  const dispatch = useDispatch();
  const [OMRSheetsData, setOMRSheetsData] = useState([]);
  const [author, setAuthor] = useState("Murali");

  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetchOMRSheets(); // fetch all OMRs
        console.log("[Dashboard] fetched OMRs raw response:", res);

        const list = res?.data ?? res;
        console.log("[Dashboard] normalized OMR list:", list);

        setOMRSheetsData(Array.isArray(list) ? list : []);
      } catch (error) {
        console.error("Error fetching OMRs:", error);
      }
    };

    fetchData();

    // When a created OMR is dispatched (optimistic or real), prepend it
    const onOMRCreated = (e) => {
      const created = e?.detail;
      if (!created) return;
      setOMRSheetsData((prev) => {
        // don't duplicate exact same _id
        if (created._id && prev.some((p) => p._id === created._id)) return prev;
        return [created, ...prev];
      });
    };

    // When server returns the real created doc, replace the temp item
    const onOMRReplace = (e) => {
      const { tempId, created } = e?.detail || {};
      if (!tempId) return;

      setOMRSheetsData((prev) => {
        // if created is null -> re-fetch to be safe (server didn't return created object)
        if (!created) {
          // try to remove temp and re-fetch
          const filtered = prev.filter((p) => p._id !== tempId);
          // re-fetch server list to get authoritative list
          (async () => {
            try {
              const r = await fetchOMRSheets();
              const l = r?.data ?? r;
              setOMRSheetsData(Array.isArray(l) ? l : filtered);
            } catch (err) {
              console.error(
                "[Dashboard] re-fetch after replace fallback failed:",
                err
              );
              setOMRSheetsData(filtered);
            }
          })();
          return prev; // return previous while async re-fetch updates state
        }

        // Replace tempId with created if temp exists, else just ensure created is at top
        const hasTemp = prev.some((p) => p._id === tempId);
        let newList;
        if (hasTemp) {
          newList = prev.map((p) => (p._id === tempId ? created : p));
        } else {
          // avoid duplicating if created already exists
          if (prev.some((p) => p._id === created._id)) return prev;
          newList = [created, ...prev];
        }
        return newList;
      });
    };

    // If optimistic create failed, remove temp card
    const onOMRCreateFailed = (e) => {
      const { tempId } = e?.detail || {};
      if (!tempId) return;
      setOMRSheetsData((prev) => prev.filter((p) => p._id !== tempId));
    };

    window.addEventListener("omrCreated", onOMRCreated);
    window.addEventListener("omrReplace", onOMRReplace);
    window.addEventListener("omrCreateFailed", onOMRCreateFailed);

    return () => {
      window.removeEventListener("omrCreated", onOMRCreated);
      window.removeEventListener("omrReplace", onOMRReplace);
      window.removeEventListener("omrCreateFailed", onOMRCreateFailed);
    };
  }, []);

  return (
    <>
      {/* --- Generated OMR Section --- */}
      <div className={styles.totalcontainer}>
        <div className={styles.dashboard}>
          <div className={styles.topBar}>
            <div className={styles.left}>
              <IconAndName
                icon={<PiExamFill size={22} />}
                title="Generated OMR's"
              />
              <SearchBar placeholder="Search by OMR name" />
            </div>
            <div className={styles.right}>
              <Button
                type="primary"
                label="Generate OMR"
                onClick={() => dispatch(openModal("generateOmr"))}
              />
            </div>
          </div>

          <div className={styles.cards}>
            {filterUserGeneratedOMR(OMRSheetsData, author).map((item) => (
              <OMRCard
                key={item._id}
                imageSrc={omrImage1}
                mcqCount={item?.numberOfQuestions}
                optionsCount={item?.numberOfOptions}
                intQuesCount={item?.numberOfIntegerQuestions}
                onView={() => navigate("/omrSheet", { state: item })}
                onDownload={() => navigate("/printOmrSheet", { state: item })}
              />
            ))}
          </div>

          <div className={styles.footer}>
            <p className={styles.viewAll}>View All</p>
          </div>
        </div>

        {/* --- Default OMR Section --- */}
        <div className={styles.dashboard}>
          <div className={styles.topBar}>
            <div className={styles.left}>
              <IconAndName
                icon={<PiExamFill size={22} />}
                title="Default OMR's"
              />
              <SearchBar placeholder="Search by OMR name" />
            </div>
          </div>

          <div className={styles.cards}>
            {OMRSheetsData.map((item) => (
              <OMRCard
                key={item._id}
                imageSrc={omrImage1}
                mcqCount={item.numberOfQuestions}
                optionsCount={item.numberOfOptions}
                intQuesCount={item.numberOfIntegerQuestions}
                onView={() => navigate("/omrSheet", { state: item })}
                onDownload={() => navigate("/printOmrSheet", { state: item })}
              />
            ))}
          </div>

          <div className={styles.footer}>
            <p className={styles.viewAll}>View All</p>
          </div>
        </div>
      </div>
    </>
  );
};

export default Dashboard;

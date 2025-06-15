import React from "react";

// Simulated files from /Dictionary (pretend these are uploaded)
const dictionaryFiles = [
  "Haida_KaiganiDictionary.json",
  "Haida_MassetDictionary.json",
  "Thlinkit_SkutkwanDictionary.json",
  "Tshimshian_KithatlāDictionary.json",
  "Tshimshian_KituntoDictionary.json",
];

// Simulated dictionaries extracted and ready
const extractedDictionaries = [
  { name: "Haida – Kaigani", file: "Haida_KaiganiDictionary.json" },
  { name: "Haida – Masset", file: "Haida_MassetDictionary.json" },
  { name: "Thlinkit – Skutkwan", file: "Thlinkit_SkutkwanDictionary.json" },
  { name: "Tshimshian – Kithatlā", file: "Tshimshian_KithatlāDictionary.json" },
  { name: "Tshimshian – Kitunto", file: "Tshimshian_KituntoDictionary.json" },
];

// Simulated ready validation & training split files
const readySplits = [
  {
    dialect: "Thlinkit_Skutkwan",
    train: "finetune_qa_Thlinkit_Skutkwan_train.jsonl",
    valid: "finetune_qa_Thlinkit_Skutkwan_valid.jsonl",
  },
  {
    dialect: "Haida_Kaigani",
    train: "finetune_qa_Haida_Kaigani_train.jsonl",
    valid: "finetune_qa_Haida_Kaigani_valid.jsonl",
  },
  {
    dialect: "Haida_Masset",
    train: "finetune_qa_Haida_Masset_train.jsonl",
    valid: "finetune_qa_Haida_Masset_valid.jsonl",
  },
  {
    dialect: "Tshimshian_Kithatlā",
    train: "finetune_qa_Tshimshian_Kithatlā_train.jsonl",
    valid: "finetune_qa_Tshimshian_Kithatlā_valid.jsonl",
  },
  {
    dialect: "Tshimshian_Kitunto",
    train: "finetune_qa_Tshimshian_Kitunto_train.jsonl",
    valid: "finetune_qa_Tshimshian_Kitunto_valid.jsonl",
  },
];

const StatusDot = ({ color = "bg-green-500" }: { color?: string }) => (
  <span
    className={`inline-block w-3 h-3 rounded-full mr-2 ${color} border-2 border-white shadow-sm align-middle`}
    aria-label="Ready"
  />
);

const FileList = ({ files }: { files: string[] }) => (
  <ul className="space-y-2">
    {files.map((f) => (
      <li
        key={f}
        className="rounded px-2 py-1 bg-muted/30 text-xs md:text-sm text-foreground border border-muted"
      >
        {f}
      </li>
    ))}
  </ul>
);

const Dashboard = () => {
  return (
    <div className="w-full h-full min-h-screen bg-background flex flex-col md:flex-row gap-4 p-2 md:p-6">
      {/* Document browser (left) */}
      <section className="flex-[1.3] min-w-[180px] max-w-[260px] bg-card border rounded-md shadow-inner flex flex-col p-4">
        <h2 className="font-semibold text-base mb-3 text-primary border-b pb-2">
          Uploaded Source Files
        </h2>
        <FileList files={dictionaryFiles} />
      </section>

      {/* Extracted dictionaries (center) */}
      <section className="flex-[2.3] min-w-[220px] bg-card border rounded-md shadow-inner flex flex-col p-4">
        <h2 className="font-semibold text-base mb-3 text-primary border-b pb-2">
          Extracted Dictionaries
        </h2>
        <ul className="space-y-3">
          {extractedDictionaries.map((dict) => (
            <li
              key={dict.file}
              className="flex items-center bg-muted/30 rounded px-2 py-1 border text-foreground"
            >
              <StatusDot />
              <span className="font-medium">{dict.name}</span>
              <span className="ml-2 text-xs text-muted-foreground">({dict.file})</span>
              <span className="ml-auto text-xs text-green-600">Ready</span>
            </li>
          ))}
        </ul>
      </section>

      {/* Ready splits (right) */}
      <section className="flex-[1.8] min-w-[200px] bg-card border rounded-md shadow-inner flex flex-col p-4">
        <h2 className="font-semibold text-base mb-3 text-primary border-b pb-2">
          Training &amp; Validation Splits
        </h2>
        <ul className="space-y-3">
          {readySplits.map((split) => (
            <li
              key={split.dialect}
              className="flex flex-col gap-0.5 bg-muted/30 rounded px-2 py-1 border"
            >
              <div className="flex items-center">
                <StatusDot />
                <span className="font-medium">{split.dialect}</span>
                <span className="ml-auto text-xs text-green-600">Ready</span>
              </div>
              <div className="pl-5 text-xs text-muted-foreground">
                train: <span className="font-mono">{split.train}</span>
              </div>
              <div className="pl-5 text-xs text-muted-foreground">
                valid: <span className="font-mono">{split.valid}</span>
              </div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
};

export default Dashboard;

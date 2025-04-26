import { useTheme } from "next-themes";
// ui
import { CircularBarSpinner } from "@utrack/ui";

export const LogoSpinner = () => {
  const { resolvedTheme } = useTheme();
  
  return (
    <div className="flex items-center justify-center">
      <CircularBarSpinner 
        width="40px" 
        height="40px" 
        className={resolvedTheme === "dark" ? "text-white" : "text-blue-700"}
      />
    </div>
  );
};

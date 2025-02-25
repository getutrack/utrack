import { Styles, StyleSheet } from "@react-pdf/renderer";
import {
  Bold,
  CaseSensitive,
  Code2,
  Heading1,
  Heading2,
  Heading3,
  Heading4,
  Heading5,
  Heading6,
  Image,
  Italic,
  List,
  ListOrdered,
  ListTodo,
  LucideIcon,
  Strikethrough,
  Table,
  TextQuote,
  Underline,
} from "lucide-react";
// editor
import { TEditorCommands, TEditorFontStyle } from "@utrack/editor";
// ui
import { MonospaceIcon, SansSerifIcon, SerifIcon } from "@utrack/ui";
// helpers
import { convertRemToPixel } from "@/helpers/common.helper";

type TEditorTypes = "lite" | "document";

export type ToolbarMenuItem = {
  key: TEditorCommands;
  name: string;
  icon: LucideIcon;
  shortcut?: string[];
  editors: TEditorTypes[];
};

export const TYPOGRAPHY_ITEMS: ToolbarMenuItem[] = [
  { key: "text", name: "Text", icon: CaseSensitive, editors: ["document"] },
  { key: "h1", name: "Heading 1", icon: Heading1, editors: ["document"] },
  { key: "h2", name: "Heading 2", icon: Heading2, editors: ["document"] },
  { key: "h3", name: "Heading 3", icon: Heading3, editors: ["document"] },
  { key: "h4", name: "Heading 4", icon: Heading4, editors: ["document"] },
  { key: "h5", name: "Heading 5", icon: Heading5, editors: ["document"] },
  { key: "h6", name: "Heading 6", icon: Heading6, editors: ["document"] },
];

const BASIC_MARK_ITEMS: ToolbarMenuItem[] = [
  { key: "bold", name: "Bold", icon: Bold, shortcut: ["Cmd", "B"], editors: ["lite", "document"] },
  { key: "italic", name: "Italic", icon: Italic, shortcut: ["Cmd", "I"], editors: ["lite", "document"] },
  { key: "underline", name: "Underline", icon: Underline, shortcut: ["Cmd", "U"], editors: ["lite", "document"] },
  {
    key: "strikethrough",
    name: "Strikethrough",
    icon: Strikethrough,
    shortcut: ["Cmd", "Shift", "S"],
    editors: ["lite", "document"],
  },
];

const LIST_ITEMS: ToolbarMenuItem[] = [
  {
    key: "bulleted-list",
    name: "Bulleted list",
    icon: List,
    shortcut: ["Cmd", "Shift", "7"],
    editors: ["lite", "document"],
  },
  {
    key: "numbered-list",
    name: "Numbered list",
    icon: ListOrdered,
    shortcut: ["Cmd", "Shift", "8"],
    editors: ["lite", "document"],
  },
  {
    key: "to-do-list",
    name: "To-do list",
    icon: ListTodo,
    shortcut: ["Cmd", "Shift", "9"],
    editors: ["lite", "document"],
  },
];

const USER_ACTION_ITEMS: ToolbarMenuItem[] = [
  { key: "quote", name: "Quote", icon: TextQuote, editors: ["lite", "document"] },
  { key: "code", name: "Code", icon: Code2, editors: ["lite", "document"] },
];

const COMPLEX_ITEMS: ToolbarMenuItem[] = [
  { key: "table", name: "Table", icon: Table, editors: ["document"] },
  { key: "image", name: "Image", icon: Image, editors: ["lite", "document"] },
];

export const TOOLBAR_ITEMS: {
  [editorType in TEditorTypes]: {
    [key: string]: ToolbarMenuItem[];
  };
} = {
  lite: {
    basic: BASIC_MARK_ITEMS.filter((item) => item.editors.includes("lite")),
    list: LIST_ITEMS.filter((item) => item.editors.includes("lite")),
    userAction: USER_ACTION_ITEMS.filter((item) => item.editors.includes("lite")),
    complex: COMPLEX_ITEMS.filter((item) => item.editors.includes("lite")),
  },
  document: {
    basic: BASIC_MARK_ITEMS.filter((item) => item.editors.includes("document")),
    list: LIST_ITEMS.filter((item) => item.editors.includes("document")),
    userAction: USER_ACTION_ITEMS.filter((item) => item.editors.includes("document")),
    complex: COMPLEX_ITEMS.filter((item) => item.editors.includes("document")),
  },
};

export const EDITOR_FONT_STYLES: {
  key: TEditorFontStyle;
  label: string;
  icon: any;
}[] = [
  {
    key: "sans-serif",
    label: "Sans serif",
    icon: SansSerifIcon,
  },
  {
    key: "serif",
    label: "Serif",
    icon: SerifIcon,
  },
  {
    key: "monospace",
    label: "Mono",
    icon: MonospaceIcon,
  },
];

const EDITOR_PDF_FONT_FAMILY_STYLES: Styles = {
  "*:not(.courier, .courier-bold)": {
    fontFamily: "Inter",
  },
  ".courier": {
    fontFamily: "Courier",
  },
  ".courier-bold": {
    fontFamily: "Courier-Bold",
  },
};

const EDITOR_PDF_TYPOGRAPHY_STYLES: Styles = {
  // page title
  "h1.page-title": {
    fontSize: convertRemToPixel(1.6),
    fontWeight: "bold",
    marginTop: 0,
    marginBottom: convertRemToPixel(2),
  },
  // headings
  "h1:not(.page-title)": {
    fontSize: convertRemToPixel(1.4),
    fontWeight: "semibold",
    marginTop: convertRemToPixel(2),
    marginBottom: convertRemToPixel(0.25),
  },
  h2: {
    fontSize: convertRemToPixel(1.2),
    fontWeight: "semibold",
    marginTop: convertRemToPixel(1.4),
    marginBottom: convertRemToPixel(0.0625),
  },
  h3: {
    fontSize: convertRemToPixel(1.1),
    fontWeight: "semibold",
    marginTop: convertRemToPixel(1),
    marginBottom: convertRemToPixel(0.0625),
  },
  h4: {
    fontSize: convertRemToPixel(1),
    fontWeight: "semibold",
    marginTop: convertRemToPixel(1),
    marginBottom: convertRemToPixel(0.0625),
  },
  h5: {
    fontSize: convertRemToPixel(0.9),
    fontWeight: "semibold",
    marginTop: convertRemToPixel(1),
    marginBottom: convertRemToPixel(0.0625),
  },
  h6: {
    fontSize: convertRemToPixel(0.8),
    fontWeight: "semibold",
    marginTop: convertRemToPixel(1),
    marginBottom: convertRemToPixel(0.0625),
  },
  // paragraph
  "p:not(table p)": {
    fontSize: convertRemToPixel(0.8),
  },
  "p:not(ol p, ul p)": {
    marginTop: convertRemToPixel(0.25),
    marginBottom: convertRemToPixel(0.0625),
  },
};

const EDITOR_PDF_LIST_STYLES: Styles = {
  "ul, ol": {
    fontSize: convertRemToPixel(0.8),
    marginHorizontal: -20,
  },
  "ol p, ul p": {
    marginVertical: 0,
  },
  "ol li, ul li": {
    marginTop: convertRemToPixel(0.45),
  },
  "ul ul, ul ol, ol ol, ol ul": {
    marginVertical: 0,
  },
  "ul[data-type='taskList']": {
    position: "relative",
  },
  "div.input-checkbox": {
    position: "absolute",
    top: convertRemToPixel(0.15),
    left: -convertRemToPixel(1.2),
    height: convertRemToPixel(0.75),
    width: convertRemToPixel(0.75),
    borderWidth: "1.5px",
    borderStyle: "solid",
    borderRadius: convertRemToPixel(0.125),
  },
  "div.input-checkbox:not(.checked)": {
    backgroundColor: "#ffffff",
    borderColor: "#171717",
  },
  "div.input-checkbox.checked": {
    backgroundColor: "#3f76ff",
    borderColor: "#3f76ff",
  },
  "ul li[data-checked='true'] p": {
    color: "#a3a3a3",
  },
};

const EDITOR_PDF_CODE_STYLES: Styles = {
  // code block
  "[data-node-type='code-block']": {
    marginVertical: convertRemToPixel(0.5),
    padding: convertRemToPixel(1),
    borderRadius: convertRemToPixel(0.5),
    backgroundColor: "#f7f7f7",
    fontSize: convertRemToPixel(0.7),
  },
  // inline code block
  "[data-node-type='inline-code-block']": {
    margin: 0,
    paddingVertical: convertRemToPixel(0.25 / 4 + 0.25 / 8),
    paddingHorizontal: convertRemToPixel(0.375),
    border: "0.5px solid #e5e5e5",
    borderRadius: convertRemToPixel(0.25),
    backgroundColor: "#e8e8e8",
    color: "#f97316",
    fontSize: convertRemToPixel(0.7),
  },
};

export const EDITOR_PDF_DOCUMENT_STYLESHEET = StyleSheet.create({
  ...EDITOR_PDF_FONT_FAMILY_STYLES,
  ...EDITOR_PDF_TYPOGRAPHY_STYLES,
  ...EDITOR_PDF_LIST_STYLES,
  ...EDITOR_PDF_CODE_STYLES,
  // quote block
  blockquote: {
    borderLeft: "3px solid gray",
    paddingLeft: convertRemToPixel(1),
    marginTop: convertRemToPixel(0.625),
    marginBottom: 0,
    marginHorizontal: 0,
  },
  // image
  img: {
    marginVertical: 0,
    borderRadius: convertRemToPixel(0.375),
  },
  // divider
  "div[data-type='horizontalRule']": {
    marginVertical: convertRemToPixel(1),
    height: 1,
    width: "100%",
    backgroundColor: "gray",
  },
  // mention block
  "[data-node-type='mention-block']": {
    margin: 0,
    color: "#3f76ff",
    backgroundColor: "#3f76ff33",
    paddingHorizontal: convertRemToPixel(0.375),
  },
  // table
  table: {
    marginTop: convertRemToPixel(0.5),
    marginBottom: convertRemToPixel(1),
    marginHorizontal: 0,
  },
  "table td": {
    padding: convertRemToPixel(0.625),
    border: "1px solid #e5e5e5",
  },
  "table p": {
    fontSize: convertRemToPixel(0.7),
  },
});

# -*- coding: utf-8 -*-
"""Internal bilingual scene registry used by data generation."""

SCENE_REGISTRY = {
    "daily_chat": {
        "name": "Daily Chat (Chinese)",
        "language": "zh",
        "noise_types": ["filler", "self_correction", "repetition", "habit"],
        "seed_categories": ["topics", "names", "places", "brands"],
    },
    "vibe_coding": {
        "name": "Vibe Coding",
        "language": "zh-en-mix",
        "noise_types": ["filler", "plan_switch", "param_correction"],
        "seed_categories": ["languages", "frameworks", "algorithms", "params", "tools"],
    },
    "explanation": {
        "name": "Entity Explanation",
        "language": "zh",
        "noise_types": ["char_explain", "negation_explain", "word_compose"],
        "seed_categories": ["ambiguous_names", "places", "surnames", "brands"],
    },
    "meeting": {
        "name": "Meeting (Chinese)",
        "language": "zh",
        "noise_types": ["data_correction", "insertion", "hesitation"],
        "seed_categories": ["departments", "metrics", "names"],
    },
    "customer_service": {
        "name": "Customer Service (Chinese)",
        "language": "zh",
        "noise_types": ["digit_correction", "address_hesitation", "confirm_repeat"],
        "seed_categories": ["product_names", "addresses", "id_formats"],
    },
    "english_daily": {
        "name": "Daily Chat (English)",
        "language": "en",
        "noise_types": ["filler", "self_correction", "hedging", "false_start"],
        "seed_categories": ["topics", "names", "places"],
    },
    "english_tech": {
        "name": "Tech (English)",
        "language": "en",
        "noise_types": ["spelling_out", "param_change", "uncertainty"],
        "seed_categories": ["cli_tools", "services", "config_params"],
    },
    "academic": {
        "name": "Academic (Chinese/Mixed)",
        "language": "zh-en-mix",
        "noise_types": ["formula_hesitation", "data_correction", "term_uncertainty"],
        "seed_categories": ["models", "datasets", "metrics", "methods"],
    },
    "navigation": {
        "name": "Navigation",
        "language": "zh",
        "noise_types": ["place_hesitation", "route_correction"],
        "seed_categories": ["cities", "stations", "roads"],
    },
    "dictation_memo": {
        "name": "Dictation Memo (Chinese/Mixed)",
        "language": "zh-en-mix",
        "noise_types": ["think_aloud", "list_hesitation", "number_correction"],
        "seed_categories": ["task_types", "people", "tools"],
    },
    "voice_search": {
        "name": "Voice Search (Chinese/Mixed)",
        "language": "zh-en-mix",
        "noise_types": ["brand_hesitation", "correction"],
        "seed_categories": ["brands", "songs", "apps"],
    },
    "english_academic": {
        "name": "English Academic",
        "language": "en",
        "noise_types": ["filler", "self_correction", "hedging", "term_uncertainty"],
        "seed_categories": ["research_topics", "metrics", "paper_references", "models"],
    },
    "english_meeting": {
        "name": "English Business Meeting",
        "language": "en",
        "noise_types": ["filler", "hedging", "false_start", "self_correction"],
        "seed_categories": ["meeting_topics", "people_names", "departments"],
    },
    "english_customer_service": {
        "name": "English Customer Service",
        "language": "en",
        "noise_types": ["filler", "repetition", "self_correction"],
        "seed_categories": ["issue_types", "order_numbers", "companies"],
    },
    "english_voice_search": {
        "name": "English Voice Search",
        "language": "en",
        "noise_types": ["correction", "hesitation_start", "mind_change"],
        "seed_categories": ["command_types", "app_names", "smart_home"],
    },
    "english_dictation": {
        "name": "English Dictation",
        "language": "en",
        "noise_types": ["think_aloud", "self_correction", "supplement", "filler"],
        "seed_categories": ["dictation_contexts", "contact_names", "time_expressions"],
    },
}

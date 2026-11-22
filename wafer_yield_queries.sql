-- ============================================================================
-- SEMICONDUCTOR ATE & WAFER YIELD SQL ANALYTICS (fab_ate_database.db)
-- Table: ate_wafer_tests (2,127 dies across 3 fabrication chambers)
-- ============================================================================

-- 1. OVERALL WAFER & CHAMBER YIELD SUMMARY (GROUP BY + CASE WHEN)
-- Calculates total dies tested, passed dies, failed dies, and Yield % per chamber
SELECT 
    Chamber,
    Wafer_ID,
    COUNT(*) AS Total_Dies_Tested,
    SUM(CASE WHEN Pass_Fail = 1 THEN 1 ELSE 0 END) AS Good_Dies,
    SUM(CASE WHEN Pass_Fail = 0 THEN 1 ELSE 0 END) AS Failed_Dies,
    ROUND(AVG(Pass_Fail) * 100.0, 2) AS Yield_Percentage,
    ROUND(AVG(Leakage_Current_mA), 2) AS Avg_Leakage_mA
FROM ate_wafer_tests
GROUP BY Chamber, Wafer_ID
ORDER BY Yield_Percentage ASC;


-- 2. EDGE-RING DEFECT ISOLATION (SPATIAL RADIAL FILTERING)
-- Compares failure rates on the outer wafer rim (Radius > 13.0) vs. inner core
SELECT 
    Chamber,
    CASE 
        WHEN Radius > 13.0 THEN 'Outer Edge Zone (R > 13)'
        ELSE 'Inner Core Zone (R <= 13)'
    END AS Wafer_Zone,
    COUNT(*) AS Die_Count,
    ROUND(AVG(Pass_Fail) * 100.0, 2) AS Zone_Yield_Pct,
    ROUND(MAX(Leakage_Current_mA), 2) AS Peak_Leakage_mA
FROM ate_wafer_tests
GROUP BY Chamber, Wafer_Zone
ORDER BY Chamber, Zone_Yield_Pct ASC;


-- 3. IDENTIFYING TOP 5 WORST-PERFORMING LEAKAGE OUTLIERS PER CHAMBER (WINDOW FUNCTION)
-- Uses RANK() OVER(PARTITION BY ...) - a classic data analyst interview question
WITH Ranked_Defects AS (
    SELECT 
        Chamber,
        Wafer_ID,
        Die_X,
        Die_Y,
        ROUND(Leakage_Current_mA, 2) AS Leakage_mA,
        ROUND(Supply_Voltage_V, 3) AS Voltage_V,
        RANK() OVER (
            PARTITION BY Chamber 
            ORDER BY Leakage_Current_mA DESC
        ) AS Leakage_Rank
    FROM ate_wafer_tests
    WHERE Pass_Fail = 0
)
SELECT *
FROM Ranked_Defects
WHERE Leakage_Rank <= 5;


-- 4. EXCURSION ALERT: CHAMBERS VIOLATING 85% MINIMUM YIELD SLA (HAVING CLAUSE)
-- Flags any fabrication chamber whose overall yield drops below the 85% threshold
SELECT 
    Chamber,
    ROUND(AVG(Pass_Fail) * 100.0, 2) AS Chamber_Yield_Pct,
    ROUND(AVG(Clock_Freq_MHz), 1) AS Avg_Clock_MHz
FROM ate_wafer_tests
GROUP BY Chamber
HAVING AVG(Pass_Fail) * 100.0 < 85.0;


-- 5. SPATIAL SCRATCH DETECTION QUERY (GEOMETRIC COORDINATE FILTER)
-- Isolates diagonal linear defects where |Die_X - Die_Y| <= 1
SELECT 
    Chamber,
    COUNT(*) AS Diagonal_Dies_Tested,
    SUM(CASE WHEN Pass_Fail = 0 THEN 1 ELSE 0 END) AS Diagonal_Failures,
    ROUND(AVG(Pass_Fail) * 100.0, 2) AS Diagonal_Zone_Yield_Pct
FROM ate_wafer_tests
WHERE ABS(Die_X - Die_Y) <= 1 AND Die_X > -6
GROUP BY Chamber;